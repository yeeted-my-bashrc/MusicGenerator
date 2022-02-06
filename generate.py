import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
import time

def generate():
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
    pitchnames = sorted(set(item for item in notes))
    nVoc = len(set(notes))
    netIn, normalizedIn = prepSequences(notes, pitchnames, nVoc)
    model = createNet(normalizedIn, nVoc)
    prediction = genNotes(model, netIn, pitchnames, nVoc)
    createMidi(prediction)

def prepSequences(notes, pitchnames, nVoc):
    toInt = dict((note, number) for number, note in enumerate(pitchnames))
    seqLen = 32
    netIn = []
    output = []
    for i in range(0, len(notes) - seqLen, 1):
        seqIn = notes[i:i + seqLen]
        seqOut = notes[i + seqLen]
        netIn.append([toInt[char] for char in seqIn])
        output.append(toInt[seqOut])
    nPatterns = len(netIn)
    normalizedIn = numpy.reshape(netIn, (nPatterns, seqLen, 1))
    normalizedIn = normalizedIn / float(nVoc)
    return (netIn, normalizedIn)

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
    model.load_weights('./data/weights/weights-99-0.1758.hdf5') # this is the last updated file, need to manually change if trained further

    return model

def genNotes(model, netIn, pitchnames, nVoc):
    start = numpy.random.randint(0, len(netIn)-1)
    toNote = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = netIn[start]
    predictionOut = []
    for noteIndex in range(200):
        predictionIn = numpy.reshape(pattern, (1, len(pattern), 1))
        predictionIn = predictionIn / float(nVoc)
        prediction = model.predict(predictionIn, verbose=0)
        index = numpy.argmax(prediction)
        result = toNote[index]
        predictionOut.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return predictionOut

def createMidi(predictionOut):
    offset = 0
    notesOut = []
    for pattern in predictionOut:
        if ('.' in pattern) or pattern.isdigit():
            chordNotes = pattern.split('.')
            notes = []
            for curNote in chordNotes:
                newNote = note.Note(int(curNote))
                newNote.storedInstrument = instrument.Piano()
                notes.append(newNote)
            newChord = chord.Chord(notes)
            newChord.offset = offset
            notesOut.append(newChord)
        else:
            newNote = note.Note(pattern)
            newNote.offset = offset
            newNote.storedInstrument = instrument.Piano()
            notesOut.append(newNote)
        offset += 0.4
    midiStream = stream.Stream(notesOut)
    midiStream.write('midi', fp=f'output/output-{time.time()}.mid') # this was a scuffed way to differentiate files, can delete the {time.time()} part

if __name__ == '__main__':
    for i in range(int(input("tracks to generate > "))):
        generate()