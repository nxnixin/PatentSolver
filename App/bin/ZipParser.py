import os
import re
import json
import zipfile
from lxml import etree
from App.bin.InputHandler import InputHandler
from App.bin.constants import DATA_INPUT
from App.bin.FiguresCleaner import FiguresCleaner
from App.bin import constants


class ZipParser(object):

    def __init__(self, folder, extension):
        self.folder = folder
        self.extension = extension
    def custom_cleaner(self, line):
        line = str(line)
        #line = line.lower()
        line = re.sub(r'PatentInspiration Url', '', line)
        line = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', line)
        line = re.sub(r'{', '(', line)
        line = re.sub(r'&quot;', '\'', line)
        line = re.sub(r'}', ')', line)
        line = re.sub(r'\t.*patentinspiration.*\n', '', line)
        line = re.sub(r'^|\n{2,}\bAbstract\b\n?', '', line)
        line = re.sub(r'^|\n{2,}\bClaims\b\n?', '', line)
        line = re.sub(r'^|\n{2,}\bDescription\b\n?', '', line)
        line = re.sub(r'fig\.', 'figure', line)
        line = re.sub(r'Fig\.', 'Figure', line)
        line = re.sub(r'FIG\.', 'Figure', line)
        line = re.sub(r'figs\.', 'figures', line)
        line = re.sub(r'FIGS\.', 'Figures', line)
        line = re.sub(r'(\w+\.)', r'\1 ', line)
        line = re.sub(r'&#39;', '\'', line)
        line = re.sub(r'&gt;', '>', line)
        line = re.sub(r'&lt;', '<', line)
        line = re.sub(r'&#176;', ' deg.', line)
        line = re.sub(r'  ', ' ', line)
        line = line.strip()
        return line

    def dataCleaner(self,line):
        with open(constants.ASSETS + "dropPart") as l:
            # next(l)
            drop_part = l.read().splitlines()
            drop_part_pattern = re.compile('|'.join(drop_part))

        line = str(line)
        #line = line.lower()
        line = re.sub(r'^([A-Z-/]+\s)+([A-Z])', r'\n\2', line)
        line = re.sub(drop_part_pattern, r'\n', line)
        line = re.sub(r'\s+\.\s?\d+\s+', ' ', line)
        line = line.strip()
        return line

    def smooth_data_cleaner(self,line):
        line = str(line)
        # line = line.lower()
        line = re.sub(r'\s+,', ',', line)
        line = re.sub(r'\d\w-\d\w (and? \d\w-\d\w)?', '', line)
        line = re.sub(r'\d\w-\d\w', '', line)
        line = re.sub(r'\(\s?(,\s?|;\s?)+\s?\)', '', line)
        line = re.sub(r'\s+\.\s\.', '.\n', line)
        line = re.sub(r'\s+\.\s+([a-z]+)', r' \1', line)
        line = re.sub(r'\s+(\.)\s+\[\s?\d+\s?]\s+', r'.\n', line)
        line = re.sub(r'\s?\[\s?\d+\s?]\s+', r'\n', line)
        line = re.sub(r'\s+(\.)\s+([A-Z]+)', r'.\n\2', line)
        line = re.sub(r'\s+;\s+', '; ', line)
        line = re.sub(r'\(\s+\'\s+\)', '', line)
        line = re.sub(r'\(\s+\)', '', line)
        line = re.sub(r'\(\s?\.\s?\)', '', line)
        line = re.sub(r'\(\s/\s?\)', '', line)
        line = re.sub(r'\s{2,}', ' ', line)
        line = re.sub(r'(\d+)\s+(\.)\s+(\d+)', r'\1.\3', line)
        line = line.strip()
        return line

    def OpenFiles(self, files):
        contentList = []
        filename =""
        for fichier in files:
            filename = os.path.basename(fichier)

            if fichier.endswith("xml"):
                doc = etree.parse(fichier)
                contentList.append(doc)
        return filename, contentList

    def openZips(self, files):
        zipLists = []
        folderpath = os.path.dirname(files[0])
        folder = os.path.basename(folderpath)
        d_folder = folderpath+"/unzipped/"
        for zips in files:
            if zipfile.is_zipfile(zips):
                zip_ref = zipfile.ZipFile(zips, 'r')
                zip_ref.extractall(d_folder)
                zip_ref.close()
                getFiles = InputHandler(d_folder,'*.xml')
                files = getFiles.get_input()
        return files

    def GetFiles(self):
        folder = self.folder
        getFiles = InputHandler(folder, '*.*')
        files = getFiles.get_input()

        filename, file_content = self.OpenFiles(self.openZips(files))
        filename = os.path.splitext(filename)[0]
        print(filename)
        count = 0
        corpus = []
        for content in file_content:
            description_list = []
            abstract_list = []
            claim_list = []
            docList = content.xpath("/QOitem/QOanswer/QOaVisu/QOdoclist")
            for doc in docList:
                doc = doc.find("./QOdocument")
                count +=1
                title_blocks = doc.xpath("./QOfield[@name='ETI']/QOpar[@num='1' and @xml:lang='EN']")
                for head in title_blocks:

                    title = head.xpath('./QOsen/descendant-or-self::*/text()')
                    pNumber = head.xpath('./@PUB')
                    Number = ' '.join(pNumber)
                    title = ' '.join(title)
                abstract_block = content.xpath("/QOitem/QOanswer/QOaVisu//QOfield[@name='EAB']/QOpar[@num='1' and @xml:lang='EN']")
                for abstract in abstract_block:
                    abstract_content = abstract.xpath('./QOsen/descendant-or-self::*/text()')
                    abstract_list.append(' '.join(abstract_content))
                    Abstracts = ' '.join(abstract_list)
                    a_abstract = self.custom_cleaner(Abstracts)
                    abstract_cleaner = FiguresCleaner(a_abstract)
                    Abstract = ' '.join(abstract_cleaner.clean_figures())



                claims_block = content.xpath("/QOitem/QOanswer/QOaVisu//QOfield[@name='CLMS']/QOpar")
                for claim in claims_block:
                    claim_content = claim.xpath('./QOsen/descendant-or-self::*/text()')
                    claim_list.append(' '.join(claim_content))
                    Claims = ' '.join(claim_list)


                description_block = content.xpath("/QOitem/QOanswer/QOaVisu//QOfield[@name='DESC']/QOpar")
                for description in description_block:
                    description_content = description.xpath('./QOsen/descendant-or-self::*/text()')
                    description_list.append(' '.join(description_content))
                    Description = ' '.join(description_list)



            values = {'filename':filename, 'title':title,'number':Number, 'abstract': Abstract, 'claims':Claims, 'description':Description}
            corpus.append(values)

        #with open(folder+"/demo.json", 'w') as json_data:
         #   json.dump(corpus, json_data)
            #print (values)
        
        return corpus

