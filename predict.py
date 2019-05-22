from predict_function import pred_args, read_json, load_checkpoint, predict
import matplotlib.pyplot as plt

# get parameters
pred_arg = pred_args()
# load a trained model
model = load_checkpoint(pred_arg.checkpoint, pred_arg.gpu)
# predict flower
predict(pred_arg.image_path, model,pred_arg.gpu, pred_arg.top_k)

