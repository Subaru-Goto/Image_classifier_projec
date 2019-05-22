# import parameters and training function
from train_function import arguments,data_loader, pre_model, training, gpu
'''
python train.py 
example: python train.py --arch 'vgg'

you may change following options below if it is required.

--data_directory (file path)
--save_dir (file path)
--arch (pre_trained model: please choose from 'vgg' or 'alexnet', default = 'vgg')
--hidden_units ( integer number, default = 2601)
--learning_rate ( float number, default = 0.001)
--epoch (integer number, default = 6)
--gpu (please choose 'cuda' or 'cpu')
'''
# set arguments
args = arguments()

print('parameters are as follows.')

print('data_directory: {}'.format(args.data_directory))
print('save_dir: {}'.format(args.save_dir))
print('selected_archtecture: {}'.format(args.arch))
print('hidden units: {}'.format(args.hidden_units))
print('learning rate: {}'.format(args.learning_rate))
print('number of epochs: {}'.format(args.epoch))
print('cuda or cpu: {}'.format(args.gpu))

print('please wait for 20 to 60 minutes for the training model...')

# model training
# load training data
train_data = data_loader(args.data_directory, 'train')
# load valid data
valid_data = data_loader(args.data_directory, 'valid')
# define which pre_trained model to use and number of hidden layer
model = pre_model(args.arch, args.hidden_units)
# choose gpu or cpu to train the model
device = gpu(args.gpu)
# adjust learning rate
learning_rate = args.learning_rate
# adjust number of epochs to run
epochs = args.epoch
# get the pretrained model archtecture vgg or alexnet
archtecture = args.arch
# get the save directory
save_dir = args.save_dir
# train the model with the above defined parameters
training(model, device, learning_rate, epochs, train_data, valid_data, archtecture, save_dir)

