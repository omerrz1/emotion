# how to use the ai models `

# Emotion Detection Model

This code loads a pretrained model for detecting emotions in text.
### Imports and Data Loading
```python

import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
```

```python

emotions = [
 'admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']

model = load_model('emotions.h5')
tokeniser = pickle.load(open('emotions_tokeniser.pickle','rb'))
```
Load the emotion labels, pretrained model, and fitted tokenizer.
Preprocess Input Text
```pytho

emotion = input('enter text: ').split(' ')

seq =tokeniser.texts_to_sequences([emotion])

padded_seq = pad_sequences(seq,maxlen=50,padding='post')
```
Take input text, tokenize it, and pad sequences for fixed model input size.
## Make Prediction
```python

predictions = model.predict(padded_seq)

print('emotion of that message is:',emotions[np.argmax(predictions)])

```