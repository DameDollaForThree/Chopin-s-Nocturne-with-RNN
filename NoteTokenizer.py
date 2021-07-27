# References:
# https://towardsdatascience.com/generate-piano-instrumental-music-by-using-deep-learning-80ac35cdbd2e

import numpy as np


class NoteTokenizer:
    """
    Class NoteTokenizer:
    - Index/Tokenize all notes with integers
    - Transform between notes and indices
    - Record the number of unique notes
    """
    
    def __init__(self):
        self.notes_to_index = {}
        self.index_to_notes = {}
        self.num_of_word = 0
        self.unique_word = 0
        self.notes_freq = {}
        
    def transform(self,list_array):
        """ Transform a list of note in string into index.

        Parameters
        ==========
        list_array : list
          list of note in string format

        Returns
        =======
        The transformed list in numpy array.

        """
        transformed_list = []
        for instance in list_array:
#             transformed_list.append([self.notes_to_index[note] for note in instance])
            transformed_list.append(int(self.notes_to_index[instance]))
        return np.array(transformed_list)

    # new version
    def partial_fit(self, tuples):
        """ Partial fit on the dictionary of the tokenizer
        each entry in the dictionary is a (note_array, duration) tuple
        
        Parameters
        ==========
        notes : list of notes
        
        """
        for curr_tuple in tuples:
            note_str = ','.join(str(a) for a in curr_tuple[0])
            new_tuple = (note_str, curr_tuple[1])
            if new_tuple in self.notes_freq:
                self.notes_freq[new_tuple] += 1
                self.num_of_word += 1
            else:
                self.notes_freq[new_tuple] = 1
                self.unique_word += 1
                self.num_of_word += 1
                self.notes_to_index[new_tuple], self.index_to_notes[self.unique_word] = self.unique_word, new_tuple
       
      # old version
#     def partial_fit(self, notes):
#         """ Partial fit on the dictionary of the tokenizer
        
#         Parameters
#         ==========
#         notes : list of notes
        
#         """
#         for note in notes:
#             note_str = ','.join(str(a) for a in note)
#             if note_str in self.notes_freq:
#                 self.notes_freq[note_str] += 1
#                 self.num_of_word += 1
#             else:
#                 self.notes_freq[note_str] = 1
#                 self.unique_word += 1
#                 self.num_of_word += 1
#                 self.notes_to_index[note_str], self.index_to_notes[self.unique_word] = self.unique_word, note_str
            
    def add_new_note(self, note):
        """ Add a new note into the dictionary

        Parameters
        ==========
        note : str
          a new note who is not in dictionary.  

        """
        assert note not in self.notes_to_index
        self.unique_word += 1
        self.notes_to_index[note], self.index_to_notes[self.unique_word] = self.unique_word, note
        
     
        
