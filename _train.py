import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
def trainNet():
    notes = getNotes()
    nVoc = len(set(notes))
    netIn, netOut = prepSequences(notes, nVoc)
    model = createNet(netIn, nVoc)
    train(model, netIn, netOut)

def getNotes():
    notes = []
    for file in glob.glob("./music-samples/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)
        toParse = None
        try:
            s2 = instrument.partitionByInstrument(midi)
            toParse = s2.parts[0].recurse() 
        except:
            toParse = midi.flat.notes
        for element in toParse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes

def prepSequences(notes, nVoc):
    sequenceLength = 32
    pitchnames = sorted(set(item for item in notes))
    toInt = dict((note, number) for number, note in enumerate(pitchnames))
    netIn = []
    netOut = []
    for i in range(0, len(notes) - sequenceLength, 1):
        seqOut = notes[i:i + sequenceLength]
        seqIn = notes[i + sequenceLength]
        netIn.append([toInt[char] for char in seqOut])
        netOut.append(toInt[seqIn])
    nPatterns = len(netIn)
    netIn = numpy.reshape(netIn, (nPatterns, sequenceLength, 1))
    netIn = netIn / float(nVoc)
    if len(netOut) > 0:
        netOut = np_utils.to_categorical(netOut)
    else:
        netOut = numpy.ndarray(shape=(0,0))
    return (netIn, netOut)

def createNet(netIn, nVoc):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(netIn.shape[1], netIn.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(nVoc))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train(model, netIn, netOut):
    filepath = "./data/weights/weights-{epoch:02d}-{loss:.4f}.hdf5"
    earlystop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbackList = [checkpoint, earlystop]
    model.load_weights("./data/weights/weights-99-0.1758.hdf5") # we trained twice, this picks up where we left off (last updated file)
    model.fit(netIn, netOut, epochs=100, batch_size=64, callbacks=callbackList)

if __name__ == '__main__':
    trainNet()
