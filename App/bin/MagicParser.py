import json
from App.bin import constants


class MagicParser(object):

    def __init__(self, jsonFile):

        self.jsonFile = jsonFile


    def get_graph(self):

        jsonFile = self.jsonFile
        with open(jsonFile) as data_file:
            data = json.load(data_file)
        return data

    def magic_parse(self):

        count_problem = 0
        count_partial_solution = 0
        count_concepts = 0
        count_parameters = 0
        parameters = []
        graph = self.get_graph(self.json_file)

        for item in graph['problem_graph']:
            count_concepts +=1
            for sub_item, value in item.items():
                if value['type'] =='partialSolution':
                    count_partial_solution +=1
                else:
                    count_problem +=1

        for item in graph['parameters']:
            for sub_item, value in item.items():
                for id, parameter in value['valeurs'].items():
                    parameters.append(parameter)
                    count_parameters += 1

        uniq_parameters_number = len(list(set(parameters)))

        return  {"concepts_number":count_concepts, "problems_number": count_problem, "partialSol_numbers":count_partial_solution, "parameters_number": count_parameters, "uniq_param_number": uniq_parameters_number}

