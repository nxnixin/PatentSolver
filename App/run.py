# -*- coding: utf-8 -*-


from App.bin import constants
from App.bin.InputHandler import InputHandler
from App.bin.PatentHandler import PatentHandler
from App.bin.CorpusProcessor import CorpusProcessor
import time

start_time = time.time()

def main():
    #renseigner nom du dossier de corpus et extension de fichier

    print("Starting process!")
    while True:
        try:
            input_folder = input("Please Enter your input folder name and press 'ENTER': ")
            # comment next line for production mode
            #input_folder= "Staubli"
            if not input_folder:
                raise ValueError("We didn't understand you.")

            files_extension = input("Please Enter your files extensions(txt,xml or * for all): ")
            #comment next line for production mode


            # original code
            # files_extension = "txt"


            # files_extension = "xml"
            if not files_extension:
                raise ValueError("We didn't understand you.")
        except ValueError as e:
            print(e)
            continue
        else:
            break

    input_folder = constants.DATA_INPUT + input_folder
    files_extension = "*." + files_extension

    iInput = InputHandler(input_folder, files_extension)
    input_data = iInput.get_input()

    pretreat_data = PatentHandler(input_data)
    clean_patent_data = pretreat_data.pretreat_data()


    process_data = CorpusProcessor(clean_patent_data,input_folder, files_extension)
    processed_data = process_data.process_corpus()

    print("Process is finished within %s seconds" % round(time.time() - start_time,2))



if __name__ == "__main__":
    main()

