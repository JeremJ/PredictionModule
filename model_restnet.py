from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from model_commons import load_data, plot_model_results
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = '/Users/jeremiaszjaworski/PycharmProjects/PredictionModule/train-experiment/'
VAL_DIR = '/Users/jeremiaszjaworski/PycharmProjects/PredictionModule/val-experiment/'
IMG_SIZE = [224, 224]

X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

resnet = ResNet50V2(input_shape=IMG_SIZE + [3], weights='imagenet', include_top=False, classes=2, pooling='max')

for layer in resnet.layers:
    layer.trainable = False
flatten = Flatten()(resnet.output)
prediction = Dense(1, activation='sigmoid')(flatten)  # recommended for binary classification
# creating a model
model = Model(inputs=resnet.input, outputs=prediction)

model.summary()
model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer=Adam(learning_rate=1e-4)
)
RANDOM_SEED = 123

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=RANDOM_SEED
)

validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)

history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=20
)

plot_model_results(model.history)

# Experiments ideas
# use not pre-processed data with image generator with settings from VGG - training acc > .98, but there is still problem with validation set, val_loss<0.6-2.8> and val_acc(~<0.67-0.9) are fluctuating
## solutions to be considerate: https://stackoverflow.com/questions/65213048/why-is-the-validation-loss-and-accuracy-oscillating-that-strong
#### Parameters to be checked
# test optimizers RMSprop/Adam/Adam(lr=1e-4)
# ResNetV2 pooling='max' or None
## 'None' give better results(better train acc ~0.98 but fluctuating val_acc ) from the begining of learning - pooling set to default
## while 'max' getting better(predictable acc and loss) in next iterations, first 10 epochs - pooling set to 'max'


# next steps: increase validation set to 500-700 images
