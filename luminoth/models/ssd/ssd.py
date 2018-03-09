import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from luminoth.models.ssd.ssd_feature_extractor import SSDFeatureExtractor
from luminoth.models.ssd.ssd_proposal import SSDProposal
from luminoth.models.ssd.ssd_target import SSDTarget
from luminoth.models.ssd.ssd_utils import (
    generate_raw_anchors, adjust_bboxes
)
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import get_saver

from luminoth.utils.bbox_transform import clip_boxes


class SSD(snt.AbstractModule):
    """TODO
    """

    def __init__(self, config, name='ssd'):
        super(SSD, self).__init__(name=name)

        # Main configuration object, it holds not only the necessary
        # information for this module but also configuration for each of the
        # different submodules.
        self._config = config

        # Total number of classes to classify.
        self._num_classes = config.model.network.num_classes

        # Turn on debug mode with returns more Tensors which can be used for
        # better visualization and (of course) debugging.
        self._debug = config.train.debug
        self._seed = config.train.seed

        # Anchor config, check out the docs of base_config.yml for a better
        # understanding of how anchors work.
        self._anchor_max_scale = config.model.anchors.max_scale
        self._anchor_min_scale = config.model.anchors.min_scale
        self._anchor_ratios = np.array(config.model.anchors.ratios)

        # Image shape (SSD has a fixed image shape)
        self.image_shape = [config.dataset.image_preprocessing.fixed_height,
                            config.dataset.image_preprocessing.fixed_width]

        # Total number of anchors per point, per endpoint.
        self._anchors_per_point = config.model.anchors.anchors_per_point

        # Weight for the localization loss
        self._loc_loss_weight = config.model.loss.localization_loss_weight

        # TODO: Why not use the default LOSSES collection?
        self._losses_collections = ['ssd_losses']

        # We want the pretrained model to be outside the ssd name scope.
        with self._enter_variable_scope():
            self.feature_extractor = SSDFeatureExtractor(
                config.model.base_network, parent_name=self.module_name
            )

    def _build(self, image, gt_boxes=None, is_training=True):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 5)`
                Where for each gt box we have (x1, y1, x2, y2, label),
                in that order.
            is_training: A boolean to whether or not it is used for training.

        Returns:
            A dictionary with the following keys:
            predictions:
            proposal_prediction: A dictionary with:
                proposals: The proposals of the network after appling some
                    filters like negative area; and NMS
                proposals_label: A tensor with the label for each proposal.
                proposals_label_prob: A tensor with the softmax probability
                    for the label of each proposal.
            bbox_offsets: A tensor with the predicted bbox_offsets
            class_scores: A tensor with the predicted classes scores
        """
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

        self.image_shape.append(3)  # Add channels to shape
        image.set_shape(self.image_shape)
        image = tf.expand_dims(image, 0)  # TODO: batch size is hardcoded to 1
        feature_maps = self.feature_extractor(image, is_training=is_training)

        # Build a MultiBox predictor on top of each feature layer and collect
        # the bounding box offsets and the category score logits they produce
        bbox_offsets_list = []
        class_scores_list = []
        for i, (feat_map_name, feat_map) in enumerate(feature_maps.items()):
            num_anchors = self._anchors_per_point[i]

            # Predict bbox offsets
            bbox_offsets_layer = slim.conv2d(
                feat_map, num_anchors * 4, [3, 3], activation_fn=None,
                scope=feat_map_name + '/conv_loc', padding='SAME'
            )
            bbox_offsets_flattened = tf.reshape(bbox_offsets_layer, [-1, 4])
            bbox_offsets_list.append(bbox_offsets_flattened)

            # Predict class scores
            class_scores_layer = slim.conv2d(
                feat_map, num_anchors * (self._num_classes + 1), [3, 3],
                activation_fn=None, scope=feat_map_name + '/conv_cls',
                padding='SAME'
            )
            class_scores_flattened = tf.reshape(class_scores_layer,
                                                [-1, self._num_classes + 1])
            class_scores_list.append(class_scores_flattened)
        bbox_offsets = tf.concat(bbox_offsets_list, axis=0)
        class_scores = tf.concat(class_scores_list, axis=0)
        class_probabilities = slim.softmax(class_scores)

        # Generate anchors (generated only once, therefore we use numpy)
        raw_anchors_per_featmap = generate_raw_anchors(
            feature_maps, self._anchor_min_scale, self._anchor_max_scale,
            self._anchor_ratios, self._anchors_per_point
        )
        anchors_list = []
        for i, (feat_map_name, feat_map) in enumerate(feature_maps.items()):
            # TODO: Anchor generation should be simpler. We should create
            #       them in image scale from the start instead of scaling
            #       them to their feature map size.
            feat_map_shape = feat_map.shape.as_list()[1:3]
            scaled_bboxes = adjust_bboxes(
                raw_anchors_per_featmap[feat_map_name], feat_map_shape[0],
                feat_map_shape[1], self.image_shape[0], self.image_shape[1]
            )
            clipped_bboxes = clip_boxes(scaled_bboxes, self.image_shape)
            anchors_list.append(clipped_bboxes)
        anchors = np.concatenate(anchors_list, axis=0)
        # TODO: They were using float64, is all this precision necesary?
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float64)

        # We'll return this dict after we fill it with predictions
        prediction_dict = {}

        # Generate targets for training
        if is_training and gt_boxes is not None:
            # Generate targets
            target_creator = SSDTarget(self._num_classes, anchors.shape[0],
                                       self._config.model.target,
                                       self._config.model.variances)
            class_targets, bbox_offsets_targets = target_creator(
                class_probabilities, anchors, gt_boxes,
                tf.cast(tf.shape(image), tf.float32)
            )

            # Filter the predictions and targets that we will ignore
            # during training due to hard negative mining.
            # We use class_targets to know which ones to ignore (they
            # are marked as -1 if they are to be ignored)
            with tf.name_scope('prepare_batch'):
                predictions_filter = tf.greater_equal(class_targets, 0)

                target_anchors = tf.boolean_mask(
                    anchors, predictions_filter)
                bbox_offsets_targets = tf.boolean_mask(
                    bbox_offsets_targets, predictions_filter)
                class_targets = tf.boolean_mask(
                    class_targets, predictions_filter)
                class_scores = tf.boolean_mask(
                    class_scores, predictions_filter)
                class_probabilities = tf.boolean_mask(
                    class_probabilities, predictions_filter)
                bbox_offsets = tf.boolean_mask(
                    bbox_offsets, predictions_filter)

            prediction_dict['target'] = {
                'cls': class_targets,
                'bbox_offsets': bbox_offsets_targets,
                'anchors': target_anchors
            }

        # Get the proposals and save the result
        proposals_creator = SSDProposal(self._num_classes,
                                        self._config.model.proposals,
                                        self._config.model.variances)
        if is_training and self._debug:
            # Generate proposals for visualization during training with debug
            proposals = proposals_creator(
                class_probabilities, bbox_offsets, target_anchors,
                tf.cast(tf.shape(image)[1:3], tf.float32)
            )
            prediction_dict['classification_prediction'] = proposals

        if not is_training:
            # Generate proposals for prediction
            proposals = proposals_creator(
                class_probabilities, bbox_offsets, anchors,
                tf.cast(tf.shape(image)[1:3], tf.float32)
            )
            prediction_dict['classification_prediction'] = proposals

        prediction_dict['cls_pred'] = class_scores
        prediction_dict['loc_pred'] = bbox_offsets

        if self._debug:
            prediction_dict['all_anchors'] = anchors
            prediction_dict['cls_prob'] = class_probabilities
            if is_training:
                prediction_dict['gt_boxes'] = gt_boxes

        return prediction_dict

    def loss(self, prediction_dict):
        """Compute the loss for SSD.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use different main keys:

                cls_pred: A dictionary with the classes classification.
                loc_pred: A dictionary with the localization predictions
                target: A dictionary with the targets for both classes and
                    localizations.

        Returns:
            A tensor for the total loss.
        """

        with tf.name_scope('losses'):

            cls_pred = prediction_dict['cls_pred']
            cls_target = tf.cast(
                prediction_dict['target']['cls'], tf.int32
            )
            # Transform to one-hot vector
            cls_target_one_hot = tf.one_hot(
                cls_target, depth=self._num_classes + 1,
                name='cls_target_one_hot'
            )

            # We get cross entropy loss of each proposal.
            # TODO: Optimization opportunity: We calculate the probabilities
            #       earlier in the program, so if we used those instead of the
            #       logits we would not have the need to do softmax here too.
            cross_entropy_per_proposal = (
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=cls_target_one_hot, logits=cls_pred
                )
            )
            # Second we need to calculate the smooth l1 loss between
            # `bbox_offsets` and `bbox_offsets_targets`.
            bbox_offsets = prediction_dict['loc_pred']
            bbox_offsets_targets = (
                prediction_dict['target']['bbox_offsets']
            )

            # We only want the non-background labels bounding boxes.
            not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
            bbox_offsets_positives = tf.boolean_mask(
                bbox_offsets, not_ignored, name='bbox_offsets_positives')
            bbox_offsets_target_positives = tf.boolean_mask(
                bbox_offsets_targets, not_ignored,
                name='bbox_offsets_target_positives'
            )

            # Calculate the smooth l1 regression loss between the flatten
            # bboxes offsets  and the labeled targets.
            reg_loss_per_proposal = smooth_l1_loss(
                bbox_offsets_positives, bbox_offsets_target_positives)

            cls_loss = tf.reduce_sum(cross_entropy_per_proposal)
            bbox_loss = tf.reduce_sum(reg_loss_per_proposal)

            # Following the paper, set loss to 0 if there are 0 bboxes
            # assigned as foreground targets.
            safety_condition = tf.not_equal(
                tf.shape(bbox_offsets_positives)[0], 0
            )
            final_loss = tf.cond(
                safety_condition,
                true_fn=lambda: (
                    (cls_loss + bbox_loss * self._loc_loss_weight) /
                    tf.cast(tf.shape(bbox_offsets_positives)[0], tf.float32)
                ),
                false_fn=lambda: 0.0
            )
            tf.losses.add_loss(final_loss)
            total_loss = tf.losses.get_total_loss()

            prediction_dict['reg_loss_per_proposal'] = reg_loss_per_proposal
            prediction_dict['cls_loss_per_proposal'] = (
                cross_entropy_per_proposal
            )

            tf.summary.scalar(
                'cls_loss', cls_loss,
                collections=self._losses_collections
            )

            tf.summary.scalar(
                'bbox_loss', bbox_loss,
                collections=self._losses_collections
            )

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )

            return total_loss

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        ssd network.
        """
        summaries = [
            tf.summary.merge_all(key=self._losses_collections[0])
        ]

        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = (
                self.feature_extractor.get_trainable_vars()
            )
            tf.logging.info('Training {} vars from pretrained module.'.format(
                len(pretrained_trainable_vars)))
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_saver(self, ignore_scope=None):
        """Get an instance of tf.train.Saver for all modules and submodules.
        """
        return get_saver((self, self.feature_extractor),
                         ignore_scope=ignore_scope)

    def load_pretrained_weights(self):
        """Get operation to load pretrained weights from file.
        """
        with tf.control_dependencies([tf.global_variables_initializer()]):
            res = self.feature_extractor.load_weights()
        return res
