# References:
# https://towardsdatascience.com/generate-piano-instrumental-music-by-using-deep-learning-80ac35cdbd2e

import numpy as np
from tqdm import tnrange, tqdm_notebook, tqdm
import random
from numpy.random import choice
import pretty_midi

from music21 import midi as midi21
from music21 import stream
import copy
import music21

import pypianoroll
from pypianoroll import StandardTrack, Multitrack, Track

# Contains functions that generate MIDI output using the trained model

def play(x):
    """Returns nothing. Outputs a midi realization of x, a note or stream.
    Primarily for use in notebooks and web environments.
    """  
    if isinstance(x, stream.Stream):
        x = copy.deepcopy(x)
        for subStream in x.recurse(streamsOnly=True, includeSelf=True):
            mss = subStream.getElementsByClass(stream.Measure)
            for ms in mss:
                ms.offset += 1.0
    if isinstance(x, music21.note.Note):
        s = stream.Stream()
        s.append(music21.note.Rest(1))
        s.append(x)
        x = s
    x.show('midi')

    
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def generate_from_random(unique_notes, seq_len=50):
    """
    Generate initial sequence of length "seq_len" all randomly.
    All notes are in index form.
    """
    generate = np.random.randint(0,unique_notes,seq_len).tolist()
    return generate
   
    
def generate_from_one_note(note_tokenizer, seq_len=50, new_notes='35'):
    """
    Generate initial sequence of length "seq_len" with last note as the passed in "new_notes".
    The last note is manually chosen.
    All notes in the front are empty notes 'e'.
    All notes are in index form.
    """
    generate = [note_tokenizer.notes_to_index[('e',1)] for i in range(seq_len-1)]
    generate += [note_tokenizer.notes_to_index[new_notes]]
    return generate


def generate_notes(generate, model, n_vocab, max_generated=1000, seq_len=50):
    """
    Generate the next "max_generated" notes using the initial sequence and the trained model
    All notes are in index form for training purpose.
    
    Inputs:
    - generate: the initial sequence of length "seq_len"
    - model: the trained model with loaded weights
    - unique_notes: the number of unique notes/tokens in our dictionary.
    - max_generated: how many notes to be generated
    - seq_len: sequence length
    """
    for i in tqdm_notebook(range(max_generated), desc='genrt'):
        # retrieve the next sequence of length "seq_len"
        test_input = np.array([generate], dtype = float)[:,i:i+seq_len]
        
        # reshape the test_input to match the input requirement of our model
        test_input = test_input.reshape((1, seq_len, 1))
        
        # normalize the test_input as we did for training
        test_input /= float(n_vocab)
        
        # predict the next note with a probability distribution as the output
        predicted_note = model.predict(test_input)    # output is a numpy array of size (1, unique_notes+1)
        
        # choose one note based on the probability distribution, (not necessary the one with the highest probability)
        random_note_pred = choice(n_vocab, 1, replace=False, p=predicted_note[0])
        
        # append the next generated note (index) to our initial piece
        generate.append(random_note_pred[0])
    return generate
    

# new version (with duration)
def write_midi_from_generated_pianoroll(note_tokenizer, generate, midi_file_name="Generated_MIDI/result.mid", start_index=49, const_tempo=50, max_generated=1000):
    """
    Convert the generated sequence to midi file using pianoroll
    """
    # convert the indices sequences to tuples sequence
    note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    
    # expand the tuple sequence to notes array
    expanded_note_string = []
    for curr_tuple in note_string:
        for i in range(curr_tuple[1]):
            expanded_note_string.append(curr_tuple[0])
    
    time_length = len(expanded_note_string)
    
    # initialize the piano roll matrix
    array_piano_roll = np.zeros((128,time_length+1), dtype=np.int16)
    
    # populate the piano roll matrix
    for index, note in enumerate(expanded_note_string[start_index:]):
        if note == 'e':
            pass
        else:
            splitted_note = note.split(',')
        for j in splitted_note:
            array_piano_roll[int(j),index] = 100    # value of pianoroll represents velocity
    # Velocity indicates how hard the key was struck when the note was played, which usually corresponds to the note's loudness
    
    # transpose it back for writing purpose
    array_piano_roll = array_piano_roll.T
    
    # initialize a tempo object
    tempo = np.zeros((time_length+1,1))
    tempo.fill(const_tempo)
    
    one_track = StandardTrack(pianoroll=array_piano_roll)
    multi_track = Multitrack(tempo=tempo, tracks=[one_track])
    pypianoroll.write(midi_file_name, multi_track)
    
    return multi_track
    
    
# old version (without duration)
# def write_midi_from_generated_pianoroll(note_tokenizer, generate, midi_file_name="Generated_MIDI/result.mid", start_index=49, const_tempo=50, max_generated=1000):
#     """
#     Convert the generated sequence to midi file using pianoroll
#     """
#     # convert the indices sequences to notes sequence
#     note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    
#     # initialize the piano roll matrix
#     array_piano_roll = np.zeros((128,max_generated+1), dtype=np.int16)
    
#     # populate the piano roll matrix
#     for index, note in enumerate(note_string[start_index:]):
#         if note == 'e':
#             pass
#         else:
#             splitted_note = note.split(',')
#         for j in splitted_note:
#             array_piano_roll[int(j),index] = 100    # value of pianoroll represents velocity
#     # Velocity indicates how hard the key was struck when the note was played, which usually corresponds to the note's loudness
    
#     # transpose it back for writing purpose
#     array_piano_roll = array_piano_roll.T
    
#     # initialize a tempo object
#     tempo = np.zeros((max_generated+1,1))
#     tempo.fill(const_tempo)
    
#     one_track = StandardTrack(pianoroll=array_piano_roll)
#     multi_track = Multitrack(tempo=tempo, tracks=[one_track])
#     pypianoroll.write(midi_file_name, multi_track)
    
#     return multi_track  
    

# this function is useless now as I no longer use the pretty midi library
def write_midi_from_generated_pretty_midi(note_tokenizer, generate, midi_file_name="Generated_MIDI/result.mid", start_index=49, fs=8, max_generated=1000):
    """
    Convert the generated sequence to midi file using pretty_midi
    Quality really really bad.
    """
    # convert the indices sequences to notes sequence
    note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    
    # initialize the piano roll matrix
    array_piano_roll = np.zeros((128,max_generated+1), dtype=np.int16)
    
    # populate the piano roll matrix
    for index, note in enumerate(note_string[start_index:]):    # exclude the 'e' in the front
        if note == 'e':
            pass
        else:
            splitted_note = note.split(',')
        for j in splitted_note:
            array_piano_roll[int(j),index] = 1
    
    # Convert a Piano Roll array into a PrettyMidi object
    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    print("Tempo {}".format(generate_to_midi.estimate_tempo()))
    
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = 100
    generate_to_midi.write(midi_file_name)
    
    
    