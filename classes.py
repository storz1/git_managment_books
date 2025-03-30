import numpy as np
import pytesseract
from pytesseract import Output

import easyocr
reader = easyocr.Reader(['en'])
import pkg_resources
from symspellpy import SymSpell, Verbosity
from nltk.metrics.distance import edit_distance
import cv2
import os
import copy
from pdf2image import convert_from_path

class Cl_Manager:
    def __init__(self):
        self.file_path = None
        self.jpg_files = []
        self.user = os.getlogin()
        self.page_number = 0
        self.image_index = 1
        self.word_index = 1
        self.current_paragraph = 0
        
    def process_new_book(self, bookname):
        self.bookname = bookname
        PDF_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/PDF_Files/' + str(self.bookname)
        JPG_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/JPG_Files/' + str(self.bookname)
        Text_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/Text_Files/' + str(self.bookname)
        Text_Numpy_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/Text_Numpy_Files/' + str(self.bookname)
        Mask_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/Mask_Files/' + str(self.bookname)
        Position_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/Position_Files/' + str(self.bookname)
        Position_Words_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/Position_Words_Files/' + str(self.bookname)
        
        if(os.path.exists(PDF_folder)):
            os.makedirs(JPG_folder, exist_ok=True)
            os.makedirs(Text_folder, exist_ok=True)
            os.makedirs(Text_Numpy_folder, exist_ok=True)
            os.makedirs(Mask_folder, exist_ok=True)
            os.makedirs(Position_folder, exist_ok=True)
            os.makedirs(Position_Words_folder, exist_ok=True)
            
        #first step: Convert pdf to jpg
        new_book_folder = 'C:/Users/' + self.user + '/Amazon_Project/Text_Recognition/New_Books/' + str(self.bookname)
        pdf_files = []
        for root, dirs, files in os.walk(PDF_folder):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if(len(pdf_files) == 1):
            pdfs = pdf_files[0]
            print(pdfs)
            poppler_path = r'C:\poppler\poppler-24.08.0\Library\bin'
            pages = convert_from_path(pdfs, 300, poppler_path=poppler_path,thread_count = 8)
            i = 1
            for page in pages:
                image_name = JPG_folder + "/" + "Page_" + str(i) + ".jpg"
                page.save(image_name, "JPEG")
                i = i+1
        
class ImageAnalysis:
    def __init__(self, image_path, custom_config=None, max_edit_distance_dictionary = 1, prefix_length=7):
        # Load dictionary
        self.max_edit_distance = 1
        self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        self.dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(self.dictionary_path, 0, 1)

        # Image variables
        self.image = cv2.imread(image_path)
        self.preprocessed_image = copy.deepcopy(self.image)

        # Image Pre Processing variables
        self.dilation_kernel_size = 5
        self.dilation_iterations = 1

        # Tesseract variables
        self.tesseract_result = None
        self.custom_config = custom_config

        # Easyocr variables
        self.easyocr_result = None

        # Text segmentation variables
        self.slender_letter = ("i", "j", "l", "t", "f", "I", "J", ".", ",", ";", ":")
        self.interpunction_marks = (".", ",", "-", ":", ";")
        self.colors = ["b", "y", "r", "g", "m", "c"]

        self.text_with_positions_tesseract = []
        self.text_with_positions_easyocr = []

        # Text matching variables
        self.matched_pairs = []
        self.hyphen_corrected_text = []
        self.corrected_text_poss = []

        # Text processing variables
        self.sorting_key = lambda term: f"{8 - term[2]}{(11 - len(str(term[1]))) * '0' + str(term[1])}"
        self.interpunction_marks = (".", ",", "-", "—", ":", ";", "'", '"')

        # GUI Variables
        self.positions_paragraphs = []
        self.mask_identical_words = []
        self.mask_hyphen_corrected_words = []
        self.mask_corrected_text_poss = []

    def process(self):
        "Function to run one image and gain text inside image"
        # Start by preprocessing image
        self.image_pre_processing()
        # Run tesseract
        self.run_tesseract()
        # Run easyocr
        self.run_easyocr()

    def run_tesseract(self):
        self.tesseract_result = []
        if self.custom_config is None:
            self.tesseract_result = pytesseract.image_to_data(self.preprocessed_image, output_type=Output.DICT, lang="eng")
        else:
            self.tesseract_result = pytesseract.image_to_data(self.preprocessed_image, output_type=Output.DICT, config=self.custom_config, lang="eng")

    def run_easyocr(self):
        self.easyocr_result = reader.readtext(self.preprocessed_image)
    

    def run_correction(self):
        self.text_with_positions_tesseract = []
        self.text_with_positions_easyocr = []
        self.matched_pairs = []
        self.mask_identical_words = []
        self.mask_corrected_text_poss = []
        self.hyphen_corrected_text = []
        self.corrected_text_poss = []
        self.paragraphs = [] #top to bottom
        self.words_per_paragraph = [] #left to right, top to bottom

        self.segment_words()
        self.word_matching()
        self.hyphen_correction()
        self.correct_text()
        self.generate_paragraph_segmentation()
        self.sort_text_into_paragraphs()
        

    def image_pre_processing(self):
        self.preprocessed_image = self.to_grayscale(self.image)
        self.preprocessed_image = self.remove_noise(self.preprocessed_image)
        self.preprocessed_image = self.thresholding(self.preprocessed_image)
    
    def sort_text_into_paragraphs(self):
        #first select text
        self.sorted_text = []
        self.sorted_mask = []
        self.sorted_text_positions = []
        self.sorted_paragraph_positions = []
        self.sorted_words_positions = []
        
        
        text = []
        borders_words = []
        
        positions_x = []
        positions_y = []
        borders = []
        mask = []
        
        
        for i in range(len(self.corrected_text_poss)):
            if(len(self.corrected_text_poss[i])==1):
                text.append(self.corrected_text_poss[i][0][0])
                borders_words.append(self.corrected_text_poss[i][0][2])
                
                positions_x.append(self.corrected_text_poss[i][0][1][0])
                positions_y.append(self.corrected_text_poss[i][0][1][1])
                borders.append(self.corrected_text_poss[i][0][2])
                mask.append(self.mask_corrected_text_poss[i])
                
            elif(len(self.corrected_text_poss[i])==2):
                text.append(self.corrected_text_poss[i][1][0])
                borders_words.append(self.corrected_text_poss[i][1][2])
                
                positions_x.append(self.corrected_text_poss[i][1][1][0])
                positions_y.append(self.corrected_text_poss[i][1][1][1])
                borders.append(self.corrected_text_poss[i][1][2])
                mask.append(self.mask_corrected_text_poss[i])
                

        #now sort them into paragraphs
        for i in range(len(self.positions_paragraphs)):
            x_start, x_ende, y_start, y_ende = self.positions_paragraphs[i]
            words_in_paragraph = []
            full_positions_in_paragraph = []
            
            borders_in_paragraph = []
            mask_in_paragraph = []
            positions_in_paragraph_x = []
            positions_in_paragraph_y = []
            
            for j in range(len(text)):
                if(positions_x[j]>x_start and positions_x[j]<x_ende and positions_y[j]>y_start and positions_y[j] < y_ende ):
                    words_in_paragraph.append(text[j])
                    full_positions_in_paragraph.append(borders_words[j])
                    
                    borders_in_paragraph.append(borders[j])
                    mask_in_paragraph.append(mask[j])
                    positions_in_paragraph_x.append(positions_x[j])
                    positions_in_paragraph_y.append(positions_y[j])
                    
            
            #now sort
            if(len(words_in_paragraph) == 0):
                continue
            
            sorted_indices = np.argsort(positions_in_paragraph_y)
            sorted_borders = [borders_in_paragraph[idx]  for idx in sorted_indices]
            sorted_words = [words_in_paragraph[idx] for idx in sorted_indices]
            sorted_full_borders = [full_positions_in_paragraph[idx] for idx in sorted_indices]
            
            sorted_mask = [mask_in_paragraph[idx] for idx in sorted_indices]
            sorted_position_x = [positions_in_paragraph_x[idx] for idx in sorted_indices]
            sorted_position_y = [positions_in_paragraph_y[idx] for idx in sorted_indices]
            
            
            
            #find lines
            final_words = []
            final_positions_words = []
            
            final_borders = []
            final_mask = []
            
            
            line_word = [sorted_words[0]]
            lines_full_borders = [sorted_full_borders[0]]
            
            lines_mask = [sorted_mask[0]]
            lines_borders = [sorted_borders[0]]
            lines_position_x = [sorted_position_x[0]]
            
            for j in range(len(sorted_position_y)-1):
                if(np.abs(sorted_position_y[j+1]-sorted_position_y[j])>20):
                    #sort in x
                    sorted_indices_x = np.argsort(lines_position_x)
                    sorted_words_x = [line_word[idx] for idx in sorted_indices_x]
                    sorted_full_borders_x = [lines_full_borders[idx] for idx in sorted_indices_x]
                    
                    sorted_borders_x = [lines_borders[idx] for idx in sorted_indices_x]
                    sorted_mask_x = [lines_mask[idx] for idx in sorted_indices_x]
                    
                    
                    for k in range(len(sorted_words_x)):
                        final_words.append(sorted_words_x[k])
                        final_positions_words.append(sorted_full_borders_x[k])
                        
                        final_mask.append(sorted_mask_x[k])
                        final_borders.append(sorted_borders_x[k])
                        
                    
                    
                    line_word = [sorted_words[j+1]]
                    lines_full_borders = [sorted_full_borders[j+1]]
                    
                    lines_mask = [sorted_mask[j+1]]
                    lines_borders = [sorted_borders[j+1]]
                    lines_position_x = [sorted_position_x[j+1]]
                    
                    
                else:
                    line_word.append(sorted_words[j+1])
                    lines_full_borders.append(sorted_full_borders[j+1])
                    
                    lines_mask.append(sorted_mask[j+1])
                    lines_borders.append(sorted_borders[j+1])
                    lines_position_x.append(sorted_position_x[j+1])
                    
           
                
            sorted_indices_x = np.argsort(lines_position_x)
            sorted_words_x = [line_word[idx] for idx in sorted_indices_x]
            sorted_full_borders_x = [lines_full_borders[idx] for idx in sorted_indices_x]
            
            sorted_borders_x = [lines_borders[idx] for idx in sorted_indices_x]
            sorted_mask_x = [lines_mask[idx] for idx in sorted_indices_x]
            
            
            for j in range(len(sorted_words_x)):
                final_words.append(sorted_words_x[j])
                final_mask.append(sorted_mask_x[j])
                final_borders.append(sorted_borders_x[j]) 
                final_positions_words.append(sorted_full_borders_x[j])
            
            self.sorted_text.append(final_words)
            self.sorted_mask.append(final_mask)
            self.sorted_text_positions.append(final_borders)
            self.sorted_paragraph_positions.append(self.positions_paragraphs[i])
            self.sorted_words_positions.append(final_positions_words)
    
    def generate_paragraph_segmentation(self):        
        #use tesseract result for generating the paragraphs 
        paragraphs = np.unique(self.tesseract_result['block_num'])
        positions_words = []
        paragraphs_position = [] #Border of Paragraph
        
        for i in range(len(paragraphs)):
            words = []
            positions_paragraph_left = []
            positions_paragraph_right = []
            positions_paragraph_top = []
            positions_paragraph_bottom = []
            positions_words = []
            words = []
            
            for j in range(len(self.tesseract_result['block_num'])):
                paragraph = self.tesseract_result['block_num'][j]
                word = self.tesseract_result['text'][j]
                if(paragraph == paragraphs[i] and word != ''):
                    
                    height = self.tesseract_result['height'][j]
                    width = self.tesseract_result['width'][j]
                    top = self.tesseract_result['top'][j]
                    left = self.tesseract_result['left'][j]
                    
                    words.append(word)
                    
                    positions_paragraph_left.append(left)
                    positions_paragraph_right.append(left+width)
                    positions_paragraph_top.append(top)
                    positions_paragraph_bottom.append(top+height)
                    positions_words.append([left,top,width,height])
            
            
            if(len(words)!=0):
                
                #calculations
                position_current_paragraph = [np.min(positions_paragraph_left), np.max(positions_paragraph_right), np.min(positions_paragraph_top), np.max(positions_paragraph_bottom)]
                paragraphs_position.append(position_current_paragraph) 
        
        self.positions_paragraphs = np.array(paragraphs_position)
    
    def correct_text(self):
        #mask: [1]:both identical and in dictionary, [1]: possible Eigenword , [2]: corrected word, [3]: general Error class
        for i in range(len(self.hyphen_corrected_text)):
            
            word_tesseract = self.hyphen_corrected_text[i][0][0]
            
            try:
                word_easyocr = self.hyphen_corrected_text[i][1][0]
            except:
                word_easyocr = ""
            
            if(len(word_tesseract)==0):
                continue
            
            if(word_tesseract == word_easyocr ):
                
                if(word_tesseract[-1] == "," or word_tesseract[-1] == "." or word_tesseract[-1]==";" or word_tesseract[-1]==":"):
                    
                    punctuation = word_tesseract[-1]
                    word_tesseract = word_tesseract[:-1]
                
                else:
                    punctuation = ""
                
                uncapitalized_word = word_tesseract.lower() 
                suggestions = self.sym_spell.lookup(uncapitalized_word, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance)
                
                best_suggestion = None
                max_count = -1

                # Iterate through suggestions
                for suggestion in suggestions:
    
                    # Check if the suggestion has an edit distance of 0
                    if suggestion.distance == 0:
                        best_suggestion = suggestion
                        break
                    # Otherwise, check if it has the highest count seen so far
                    elif suggestion.count > max_count:
                        best_suggestion = suggestion
                        max_count = suggestion.count
                        
                if(best_suggestion is None ):
                    self.mask_corrected_text_poss.append(1)
                    self.corrected_text_poss.append(self.hyphen_corrected_text[i])
                elif(uncapitalized_word == best_suggestion.term):
                    self.corrected_text_poss.append(self.hyphen_corrected_text[i])
                    self.mask_corrected_text_poss.append(0)
                else:
                    self.mask_corrected_text_poss.append(2)
                    new_input = self.hyphen_corrected_text[i]
                    if(word_tesseract == uncapitalized_word):
                        new_input[0][0] = best_suggestion.term + punctuation
                        new_input[1][0] = best_suggestion.term + punctuation
                    elif(len(word_tesseract) == len(best_suggestion.term)):
                        new_input_word = ""
                        for i in range(len(word_tesseract)):
                            if(word_tesseract[i].lower()==word_tesseract[i]):
                                new_input_word += best_suggestion.term[i]
                            else:
                                new_input_word += best_suggestion.term[i].upper()
                        new_input_word += punctuation
                        new_input[0][0] = new_input_word
                        new_input[1][0] = new_input_word
                    else:
                        "digit added or removed"
                        new_input = self.hyphen_corrected_text[i]
                        new_input_word = ""
                        #chek if whole word is upper case:
                        if(word_tesseract == word_tesseract.upper()):
                            new_input_word = best_suggestion.term.upper()
                        else:
                            for i in range(len(best_suggestion.term)):
                                if(i==0):
                                    new_input_word += best_suggestion.term[i].upper()
                                else:
                                    new_input_word += best_suggestion.term[i]
                                
                        new_input_word += punctuation
                        new_input[0][0] = new_input_word
                        new_input[1][0] = new_input_word
                        
                    self.corrected_text_poss.append(new_input)
            
            elif(word_easyocr == ""):
                
                if(word_tesseract[-1] == "," or word_tesseract[-1] == "." or word_tesseract[-1]==";" or word_tesseract[-1]==":"):
                    
                    punctuation = word_tesseract[-1]
                    word_tesseract = word_tesseract[:-1]
                
                else:
                    punctuation = ""
                
                #only tesseract word
                uncapitalized_word = word_tesseract.lower()
                suggestions = self.sym_spell.lookup(uncapitalized_word, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance)
                
                best_suggestion = None
                max_count = -1
                # Iterate through suggestions
                for suggestion in suggestions:
    
                    # Check if the suggestion has an edit distance of 0
                    if suggestion.distance == 0:
                        best_suggestion = suggestion
                        break
                    # Otherwise, check if it has the highest count seen so far
                    elif suggestion.count > max_count:
                        best_suggestion = suggestion
                        max_count = suggestion.count
                        
                
                
                
                if(best_suggestion is None):
                    self.mask_corrected_text_poss.append(3)
                    self.corrected_text_poss.append(self.hyphen_corrected_text[i])
                elif(uncapitalized_word == best_suggestion.term):
                    self.corrected_text_poss.append(self.hyphen_corrected_text[i])
                    self.mask_corrected_text_poss.append(3)
                else:
                    self.mask_corrected_text_poss.append(3)
                    new_input = self.hyphen_corrected_text[i]
                    if(word_tesseract == uncapitalized_word):
                        new_input[0][0] = best_suggestion.term + punctuation
                    elif(len(word_tesseract) == len(best_suggestion.term)):
                        new_input_word = ""
                        for i in range(len(word_tesseract)):
                            if(word_tesseract[i].lower()==word_tesseract[i]):
                                new_input_word += best_suggestion.term[i]
                            else:
                                new_input_word += best_suggestion.term[i].upper()
                        new_input_word += punctuation
                        new_input[0][0] = new_input_word
                    else:
                        "digit added or removed"
                        new_input_word = ""
                        #chek if whole word is upper case:
                        if(word_tesseract == word_tesseract.upper()):
                            new_input_word = best_suggestion.term.upper()
                        else:
                            for i in range(len(best_suggestion.term)):
                                if(i==0):
                                    new_input_word += best_suggestion.term[i].upper()
                                else:
                                    new_input_word += best_suggestion.term[i]
                        
                        new_input = self.hyphen_corrected_text[i]
                        new_input_word += punctuation
                        new_input[0][0] = new_input_word
                        
                    self.corrected_text_poss.append(new_input)
                
            else:
            #words different:->Trust easyocr more
                if(word_easyocr[-1] == "," or word_easyocr[-1] == "." or word_easyocr[-1]==";" or word_easyocr[-1]==":"):
                    
                    punctuation = word_easyocr[-1]
                    word_easyocr = word_easyocr[:-1]
                
                else:
                    punctuation = ""
                
                uncapitalized_word = word_easyocr.lower() 
                suggestions = self.sym_spell.lookup(uncapitalized_word, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance)
                
                best_suggestion = None
                max_count = -1
                # Iterate through suggestions
                for suggestion in suggestions:
    
                    # Check if the suggestion has an edit distance of 0
                    if suggestion.distance == 0:
                        best_suggestion = suggestion
                        break
                    # Otherwise, check if it has the highest count seen so far
                    elif suggestion.count > max_count:
                        best_suggestion = suggestion
                        max_count = suggestion.count
                        
                
                
                
                if(best_suggestion == None):
                    self.mask_corrected_text_poss.append(1)
                    self.corrected_text_poss.append(self.hyphen_corrected_text[i])
                elif(uncapitalized_word == best_suggestion.term):
                    self.corrected_text_poss.append(self.hyphen_corrected_text[i])
                    self.mask_corrected_text_poss.append(0)
                else:
                    self.mask_corrected_text_poss.append(2)
                    new_input = self.hyphen_corrected_text[i]
                    if(word_easyocr == uncapitalized_word):
                        new_input[0][0] = best_suggestion.term + punctuation
                    elif(len(word_easyocr) == len(best_suggestion.term)):
                        new_input_word = ""
                        for i in range(len(word_easyocr)):
                            if(word_easyocr[i].lower()==word_easyocr[i]):
                                new_input_word += best_suggestion.term[i]
                            else:
                                new_input_word += best_suggestion.term[i].upper()
                        new_input_word += punctuation
                        new_input[0][0] = new_input_word
                    else:
                        "digit added or removed"
                        new_input_word = ""
                        #chek if whole word is upper case:
                        if(word_easyocr == word_easyocr.upper()):
                            new_input_word = best_suggestion.term.upper()
                        else:
                            for i in range(len(best_suggestion.term)):
                                if(i==0):
                                    new_input_word += best_suggestion.term[i].upper()
                                else:
                                    new_input_word += best_suggestion.term[i]
                        
                        new_input = self.hyphen_corrected_text[i]
                        new_input_word += punctuation
                        new_input[0][0] = new_input_word
                        
                    self.corrected_text_poss.append(new_input)
            
                    

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def dilate(self, image):
        kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=self.dilation_iterations)

    def remove_noise(self, image):
        return cv2.medianBlur(image, 1)

    def segment_words(self):
        words = []
        n_boxes = len(self.tesseract_result['text'])
        for i in range(n_boxes):
            if not self.tesseract_result['word_num'][i]:
                continue
            (x, y, w, h) = (self.tesseract_result['left'][i], self.tesseract_result['top'][i],
                            self.tesseract_result['width'][i], self.tesseract_result['height'][i])
            text = self.tesseract_result['text'][i]
            center = x + w // 2, y + h // 2
            rect = [x, y, w, h]
            words.append([text, center, rect])
        self.text_with_positions_tesseract = words

        words = []
        for (box, text, confidence) in self.easyocr_result:
            (top_l, top_r, bottom_r, bottom_l) = box
            x, y = top_l
            w, h = bottom_r[0] - top_l[0], bottom_r[1] - top_l[1]
            box = [x, y, w, h]
            for word, sub_box in zip(*self.split_into_words(text, box)):
                x, y, w, h = sub_box
                text = word
                rect = [x, y, w, h]
                center = x + w // 2, y + h // 2
                words.append([text, center, rect])
        self.text_with_positions_easyocr = words

    def word_matching(self):
        self.matched_pairs = []
        main_word_list = self.text_with_positions_tesseract
        secondary_word_lists = self.text_with_positions_easyocr
        len_main = len(main_word_list)
        len_secondary = len(secondary_word_lists)
        bidir_search_radius = 5

        for index, word_pos in enumerate(main_word_list):
            word, center, rect = word_pos
            x, y, w, h = rect
            center_x, center_y = center

            left_bound = len_secondary / len_main * (index - bidir_search_radius)
            right_bound = len_secondary / len_main * (index + bidir_search_radius)

            left_bound = max(0, int(left_bound))
            right_bound = min(int(right_bound), len_secondary)

            center_dists = []
            for match_index in range(left_bound, right_bound):
                match_candidate = secondary_word_lists[match_index]
                match_word, match_center, match_rect = match_candidate
                match_center_x, match_center_y = match_center

                center_dist = ((center_x - match_center_x) ** 2
                               + (center_y - match_center_y) ** 2)

                if center_dist > h ** 2:
                    center_dists.append(-1)
                elif edit_distance(match_word, word) > max(1, 0.6 * len(word)):
                    center_dists.append(-1)
                else:
                    center_dists.append(center_dist)

            positive_center_dists = [dist for dist in center_dists if dist >= 0]
            if positive_center_dists:
                min_index = center_dists.index(min(positive_center_dists)) + left_bound
                best_match_candidate = secondary_word_lists[min_index]
                self.matched_pairs.append([word_pos, best_match_candidate])
                if word == best_match_candidate[0]:
                    self.mask_identical_words.append(1)
                else:
                    self.mask_identical_words.append(0)
            else:
                self.matched_pairs.append([word_pos])
                self.mask_identical_words.append(2)
            

    def word_width_estimate(self, word):
        result = len(word)
        for slender_letter in self.slender_letter:
            result -= 0.4 * word.count(slender_letter)
        return result

    def split_into_words(self, text, box):
        split_text = [word for word in text.split(" ") if word]
        split_text2 = split_text
        word_count = len(split_text2)
        character_sum = sum([self.word_width_estimate(word) for word in split_text2]) + ((word_count - 1) if word_count > 1 else 0)
        boxes = []
        x, y, w, h = box
        sub_box_y = y
        sub_box_h = h
        if(character_sum != 0):
            char_width = w / character_sum
            sub_box_x = x
            for index, word in enumerate(split_text2):
                sub_box_w = char_width * self.word_width_estimate(word)
                boxes.append([sub_box_x, sub_box_y, int(sub_box_w), sub_box_h])
                sub_box_x += sub_box_w + char_width
        else:
            return [],[]
            
        return split_text2, boxes

    def word_lookup(self, *choices, wrong_percentage=0.2, max_edit_dist=2):
        if all(map(lambda x: x.isdigit(), choices)):
            return [[choice, 0, 0] for choice in choices]

        suggestions = []
        for term in choices:
            if not term:
                continue

            interpunction = ""
            if term[-1] in self.interpunction_marks:
                interpunction = term[-1]
                term = term[:-1]

            edit_dist = min(max_edit_dist, ceil(wrong_percentage * len(term)))

            lookup = self.sym_spell.lookup(
                term, Verbosity.CLOSEST, max_edit_distance=edit_dist,
                transfer_casing=True, include_unknown=True,
                )

            suggestions += [
                [suggestion.term + interpunction, suggestion.count, suggestion.distance] 
                for suggestion in lookup
                ]

        if len(suggestions) < 2:
            return suggestions

        sorted_suggestions = sorted(suggestions, key=self.sorting_key, reverse=True)

        if sorted_suggestions[0][0] == sorted_suggestions[1][0]:
            return [sorted_suggestions[0]]

        for index, word in enumerate(sorted_suggestions):
            for compare_word in sorted_suggestions[index + 1:]:
                if word[0] == compare_word[0]:
                    sorted_suggestions[index][0] += "§"
                    sorted_suggestions[index][2] -= 1
                    break

        return sorted(sorted_suggestions, key=self.sorting_key, reverse=True)



    def hyphen_correction(self):

        self.hyphen_corrected_text = []
        self.mask_hyphen_corrected_words = []

        index = 0
        while index < len(self.matched_pairs)-1 :

            word_pos      = self.matched_pairs[index  ][0]
            next_word_pos = self.matched_pairs[index+1][0]

            word      = word_pos[0]
            next_word = next_word_pos[0]

            center      = word_pos[1]
            next_center = next_word_pos[1]

            x,next_x = center[0],next_center[0]


            if word[-1] in ("-",) and next_x < x:


                joined_word = self.matched_pairs[index]
                # join word for all choices
                # calculate rect anew (width) but keep center for all choices
                for i in range(len(joined_word)):
                    if i < len(self.matched_pairs[index]) and i < len(self.matched_pairs[index+1]):
                        joined_word[i][0]    = self.matched_pairs[index][i][0][:-1] + self.matched_pairs[index+1][i][0]
                        joined_word[i][2][2] = self.matched_pairs[index][i][2][2] + self.matched_pairs[index+1][i][2][2]

                #print(text[index],index)
                self.hyphen_corrected_text.append(joined_word)
                self.mask_hyphen_corrected_words.append(3)

                index += 2

            else:
                self.hyphen_corrected_text.append(self.matched_pairs[index])
                self.mask_hyphen_corrected_words.append(self.mask_identical_words[index])

                index += 1

        if index < len(self.matched_pairs):
            self.hyphen_corrected_text.append(self.matched_pairs[index])
            self.mask_hyphen_corrected_words.append(self.mask_identical_words[index])

        
    