# -*- coding: utf-8 -*-

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port 8080
import glob
import os


class InputHandler(object):

    def __init__(self, folder_path, extension):
        self.folder_path = folder_path
        self.extension = extension

        print("Handling Corpus...")


    def _get_dirs(self, base):
        return [x for x in glob.iglob(os.path.join(base, '*')) if os.path.isdir(x)]

    def get_base_file(self, base, pattern):
        lList = []
        lList.extend(glob.glob(os.path.join(base, pattern)))
        dirs = self._get_dirs(base)
        if len(dirs):
            for d in dirs:
                lList.extend(self.get_base_file(os.path.join(base, d), pattern))
        return lList

    def get_input(self):
        folder_path = self.folder_path
        extension = self.extension
        patent_files = self.get_base_file(folder_path, extension)
        return patent_files


