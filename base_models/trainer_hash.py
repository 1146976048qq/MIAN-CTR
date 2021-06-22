from alps.common.tools import BaseTrainer
import time
import logging


class TrainerHash(BaseTrainer):

    def before_run(self):
        super(TrainerHash, self).before_run()
        # 判断是否使用了dense特征
        self.config.use_dense = False
        for m in self.config.x:
            if 'dense' in m['type']:
                self.config.use_dense = True

    def prepare_model_and_signature(self, inputs_tensor=None, labels_tensor=None):
        ret = super(TrainerHash,self).prepare_model_and_signature(inputs_tensor, labels_tensor)
        inputs_layer_util = self._model_inputs_ctx
        inputs_layer_util.inference_signature['use_feature_map'] = False
        self.signature_json = inputs_layer_util.get_inference_signature_json()

        return ret
