# part4.py - Model Architecture and Training using HDF5 and Generators

from part3 import train_generator, val_generator, class_labels
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ Constants
IMG_SIZE = 224
SEQUENCE_LENGTH = 4
LR = 1e-4
EPOCHS = 50

# ✅ Model Input Shape (match generator output)
input_shape = (SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
input_layer = Input(shape=input_shape, name='input_layer')

# ✅ MobileNetV2 as feature extractor
mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
mobilenet.trainable = False

# Apply MobileNetV2 on each frame in the sequence
x = TimeDistributed(mobilenet)(input_layer)
x = TimeDistributed(GlobalAveragePooling2D())(x)

# LSTM for temporal sequence learning
x = LSTM(128, return_sequences=False)(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)

# Output layer
output = Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ✅ Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# ✅ Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ✅ Save Model
model.save("skin_cancer_cnn_lstm_model.keras")
print("✅ Model training complete and saved as skin_cancer_cnn_lstm_model.keras")
