# test for correct syntax for file_writing
# Syntax shold be like so:

import sng_parser
from complete_py_pipeline import scenegraphTable

sentence = "A small peacock sits on a large gray boulder. The peacock also sits next to a larger bird. The boulder sits in a wide river. It also has small pebbles around it."
graph = scenegraphTable(sentence)
# print(graph)


# writing format -> get entity ID from list(graph['entities'].keys())[i]
# Read relations before writing: For each relation, read entity ID, resolve by above functions
#   For relations[i]['subject'], push to stack => sbj<relationItr>_<assetName>_<entityID>

#   Next push the relations[i]['relation']
#   Next push to

def getAssetEntityPair():
    asset_entity_dict = dict()


# def setSubjObjDict(graph):

#     return sub_obj_dict

def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5
