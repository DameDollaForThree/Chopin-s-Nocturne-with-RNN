# References:
# https://towardsdatascience.com/generate-piano-instrumental-music-by-using-deep-learning-80ac35cdbd2e

import numpy as np
from tqdm import tnrange, tqdm_notebook, tqdm
import pretty_midi
import glob
import pypianoroll
from pypianoroll import StandardTrack

# Contains functions that preprocess the input data


def midi_to_piano_rolls(midi_files):
    """
    Read in input MIDI files, extract their piano rolls
    Using the pypianoroll library!!
    The pretty_midi library is really bad.
    
    Input: 
    - Path to input MIDI files
    - Sampling frequency of the piano rolls
    
    Return:
    - a dictionary that stores {"file_name": piano_rolls}
    """
    pieces_rolls_dict = {}
    for file in glob.glob(midi_files):
        # extract all tracks from one piece of music
        myTracks = pypianoroll.read(file)
        all_tracks = myTracks.tracks    # an array
        
        # initialize the curr_pianoroll with the first track
        curr_pianoroll = all_tracks[0].pianoroll
        
        # if there are other tracks, simply add them all up.
        for i in range(1, len(all_tracks)):
            curr_pianoroll += all_tracks[i].pianoroll
            
        # transpose it for later use
        curr_pianoroll = np.array(curr_pianoroll).T
        pieces_rolls_dict[file] = curr_pianoroll
    return pieces_rolls_dict



def piano_rolls_to_times_notes_dict(pieces_rolls_dict):
    """
    Read in piano rolls, and extract their times & notes properties 
    Represent what note is played at what time.
    
    Input: 
    - A dictionary that stores the piano rolls
    
    Return:
    - A list of dictionaries that stores (times, notes) of each piece
    """
    times_notes_dict_list = []
    for piano_roll in pieces_rolls_dict.values():
        # process each piano roll into a time & notes dictionary
        tuple = np.where(piano_roll > 0)    # (notes, time) tuple, both values > 0
        all_notes = tuple[0]
        all_times = tuple[1]
        unique_times = np.unique(all_times)

        times_notes_dict = {}
        for curr_unique_time in unique_times:
            index_where = np.where(all_times == curr_unique_time)
            notes = all_notes[index_where]
            times_notes_dict[curr_unique_time] = notes

        times_notes_dict_list.append(times_notes_dict)
    return times_notes_dict_list


def add_empty_note_to_dict(times_notes_dict_list):
    """
    Fill the empty time slot with string 'e'
    """
    new_list = []
    for times_notes_dict in times_notes_dict_list:
        for i in range(list(times_notes_dict.keys())[0], list(times_notes_dict.keys())[-1]):
            if i in times_notes_dict:
                pass
            else:
                times_notes_dict[i] = 'e'
        new_list.append(times_notes_dict)
    return new_list


def encode_notes_dict_with_duration(times_notes_dict_list):
    """
    Encode the times_notes_dict_list with duration by combining a sequence of same notes into one tuple.
    
    Input:
    - A list of dictionaries that stores (times, notes) of each piece
    
    Return:
    - A list of dictionaries that stores {times: (notes, duration)} of each piece
    """
    new_list = []
    for times_notes_dict in times_notes_dict_list:
        new_dict = {}

        i = list(times_notes_dict.keys())[0]
        end = list(times_notes_dict.keys())[-1]
        while True:
            if i > end:
                break;
            curr_note = times_notes_dict[i]
            counter = 1
            for j in range (i+1, end):
                if np.array_equal(times_notes_dict[j], curr_note):
                    counter += 1
                else:
                    break
            # ignore extremely short notes
#             if counter > 2:
            new_dict[i] = (curr_note, counter)
            i += counter
            
        new_list.append(new_dict)
    return new_list


# new version (encode with duration)
def generate_input_and_target(times_notes_dict, seq_len=50):
    """ Generate input and the target of our deep learning for one music. Not tokenized yet.
    
    Parameters
    ==========
    times_notes_dict : dict
      Dictionary of timestep and notes
    seq_len : int
      The length of the sequence
      
    Returns
    =======
    Tuple of list of input and list of target of neural network.
       
    """
    
    # Get the start time and end time
    start_time, end_time = list(times_notes_dict.keys())[0], list(times_notes_dict.keys())[-1]
    
    # make the times_notes_dict an array for sliding window purpose
    notes_tuple_array = []
    for i in range(start_time, end_time+1):
        if i in times_notes_dict:
            notes_tuple_array.append(times_notes_dict[i])
    
    list_training, list_target = [], []
    
    for window_index in range(len(notes_tuple_array) - seq_len - 1):
        
        # initialize current traing list and target list
        list_append_training, list_append_target = [], []
        start_iterate = 0
        flag_target_append = False # flag to append the test list
        
        # pad 'e' in the front for the first "seq_len" sequences
        if window_index < seq_len - 1:
            start_iterate = seq_len - window_index - 1
            for i in range(start_iterate): 
                list_append_training.append(('e',1))
                flag_target_append = True

            # append the following tuples to the current training list
            remain_tuples_num = seq_len - start_iterate
            for j in range(remain_tuples_num):
                next_tuple = notes_tuple_array[j]
                next_tuple_str = (','.join(str(x) for x in next_tuple[0]), next_tuple[1])
                list_append_training.append(next_tuple_str)
            target_tuple = notes_tuple_array[remain_tuples_num]
                
        elif window_index >= seq_len - 1:
            for j in range(window_index - seq_len + 1, window_index - seq_len + 11):
                next_tuple = notes_tuple_array[j]
                next_tuple_str = (','.join(str(x) for x in next_tuple[0]), next_tuple[1])
                list_append_training.append(next_tuple_str)
            target_tuple = notes_tuple_array[window_index - seq_len + 11]

        # add the next target tuple to the list_append_target
        target_tuple_str = (','.join(str(x) for x in target_tuple[0]), target_tuple[1])
        list_append_target.append(target_tuple_str)
        
        list_training.append(list_append_training)
        list_target.append(list_append_target)
       
    return list_training, list_target


# old version (without duration)
# def generate_input_and_target(dict_keys_time, seq_len=50):
#     """ Generate input and the target of our deep learning for one music.
    
#     Parameters
#     ==========
#     dict_keys_time : dict
#       Dictionary of timestep and notes
#     seq_len : int
#       The length of the sequence
      
#     Returns
#     =======
#     Tuple of list of input and list of target of neural network.
       
#     """
#     # Get the start time and end time
#     start_time, end_time = list(dict_keys_time.keys())[0], list(dict_keys_time.keys())[-1]
#     list_training, list_target = [], []
#     for index_enum, time in enumerate(range(start_time, end_time)):
#         list_append_training, list_append_target = [], []
#         start_iterate = 0
#         flag_target_append = False # flag to append the test list
#         if index_enum < seq_len:
#             start_iterate = seq_len - index_enum - 1
#             for i in range(start_iterate): # add 'e' to the seq list. 
#                 list_append_training.append('e')
#                 flag_target_append = True

#         for i in range(start_iterate,seq_len):
#             index_enum = time - (seq_len - i - 1)
#             if index_enum in dict_keys_time:
#                 list_append_training.append(','.join(str(x) for x in dict_keys_time[index_enum]))      
#             else:
#                 list_append_training.append('e')

#         # add time + 1 to the list_append_target
#         if time+1 in dict_keys_time:
#             list_append_target.append(','.join(str(x) for x in dict_keys_time[time+1]))
#         else:
#             list_append_target.append('e')
#         list_training.append(list_append_training)
#         list_target.append(list_append_target)
#     return list_training, list_target

 
