from keras import Model, layers, Input

text_vocabulary_size = 10000
question_vocablary_size = 10000
answer_vocabulary_size = 500

## For Text Input
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size)(text_input, 64)
encoded_text = layers.LSTM(32)(embedded_text)

#
## For Question Input
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocablary_size, 32,output_dim=1)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

## Merge 2 Inputs
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

## Classfier
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics = ['acc'])
