import sng_parser
import re
from sentence_transformers import SentenceTransformer, util
import torch
from rdflib.namespace import FOAF, XMLNS, XSD, RDF, RDFS, OWL
import rdflib.plugins.sparql as sparql
from rdflib import Graph, URIRef, Literal, BNode, Namespace
import networkx as nx
import matplotlib.pyplot as plt
import sng_parser

# from time import time


def new_asset_desc_matching():
    from nltk.corpus import stopwords
    from nltk import download
    download('stopwords')
    stop_words = stopwords.words('english')
    sentence_a = [w for w in sentence_a if w not in stop_words]
    sentence_b = [w for w in sentence_b if w not in stop_words]

    start = time()
    import os
    from genism.models import Word2Vec
    distance = model.wmdistance(sentence_a, sentence_b)
    print("Cell took %.2f seconds to run." % (time() - start))


N_id = 1


jar_path = 'stanford-corenlp-4.5.1/stanford-corenlp-4.5.1.jar'
models_jar_path = 'stanford-corenlp-4.5.1/stanford-corenlp-4.5.1-models.jar'


def formatAssetWriting(entity_asset_dict, subject_all_objects_dict):
    formattedDict = dict()
    # In the new dict, store in subject_all_objects_dict values, but format its keys to
    # AssetName_AssetID (entity_asset_dict.keys) +_+<entityID> (subject_all_objects_dict.keys)
    # Compare - from subject_all_objects_dict, remove from keys and value tuples, everything after & including the last '_'
    #         - replace remaining keys & values in subject_all_objects_dict with values of keys that match with them

    # EXAMPLE
    # Entity Asset Dictionary:
    # {'Cow': 'Wooden_Chair_uknkaffaw', 'sidewalk': 'Brick_Wall_vgvmcgf', 'city area': 'Brick_Wall_vgvmcgf', 'shops': 'Wooden_Chair_uknkaffaw'}

    # All Subject-Objects Dictionary:
    #  {'Cow_0': [('on', 'sidewalk_1'), ('in', 'city area_2')], 'city area_2': [('near', 'shops_3')]}

    # RESULT
    # {'Wooden_Chair_uknkaffaw_0': [('on', 'Brick_Wall_vgvmcgf_1'), ('in', 'Brick_Wall_vgvmcgf_2')], 'Brick_Wall_vgvmcgf_2': [('near', 'Wooden_Chair_uknkaffaw_3')]}

    for k, v in subject_all_objects_dict.items():
        formattedKey = k
        formattedVal = v
        if k.replace(k[k.rfind('_'):], '') in list(entity_asset_dict.keys()):
            formattedKey = entity_asset_dict[k.replace(
                k[k.rfind('_'):], '')] + k[k.rfind('_'):]

        for values in v:
            prep, val = list(values)
            if val.replace(val[val.rfind('_'):], '') in list(entity_asset_dict.keys()):
                formattedVal = (prep, entity_asset_dict[val.replace(
                    val[val.rfind('_'):], '')] + val[val.rfind('_'):])

            try:
                # case: if key exists
                check = formattedDict[formattedKey]
                formattedDict[formattedKey].append(formattedVal)

            except:
                # case: if key is not yet added to dictionary
                formattedDict[formattedKey] = [formattedVal]

    return formattedDict


def subjectAllObjectsDict(scenegraph):
    # function to take in scene graph, return dictionary with key being the subject+'_'+uniqueEntityID from relations_list, and value being a list of all the objects and their relations with the subject
    # e.g. A green leaf sits on a gray boulder and under a pebble.
    #      {'leaf_0': [(on, boulder_1), (under, pebble_2)]}

    objects = []
    entities_list = scenegraph["entities"]
    for entity in entities_list:
        objects.append(entity["head"])
    relations_list = scenegraph["relations"]

    print('Printing relations: ', '\n')
    subject_objects_dict = dict()
    for relation in relations_list:
        # subject+'_'+uniqueEntityID
        subject = objects[relation.get("subject")]+'_'+str(relation['subject'])
        relation_object = (
            relation['relation'], objects[relation.get("object")]+'_'+str(relation['object']))

        try:
            # case: if same subject exists (need to be made possible after co-reference resolution)
            # print("Subject exists with object ",
            #       subject_objects_dict[subject], "already!")
            check = subject_objects_dict[subject]
            subject_objects_dict[subject].append(relation_object)

        except:                                     # case: if subject is not yet added to dictionary
            # print("Subject", subject, " is now being created.")
            subject_objects_dict[subject] = [relation_object]
    return subject_objects_dict


def scenegraphGraph(scenegraph):
    G = nx.DiGraph()
    objects = []
    similarity_input = {}

    # HEAD
    entities_list = scenegraph["entities"]
    for entity in entities_list:
        val = entity["head"]
        similarity_input[val] = []
        objects.append(val)
        G.add_node(val)

    # MODIFIERS
    for entity in entities_list:
        for modifier_dict in entity["modifiers"]:
            modifier = modifier_dict.get("span")
            G.add_node(modifier)
            object = entity["head"]
            G.add_edge(object, modifier)
            # if 'a' not in modifier and 'A' not in modifier and 'the' not in modifier and 'The' not in modifier:
            similarity_input[object].append(modifier)

    # RELATIONS
    relations_list = scenegraph["relations"]
    print('Printing relations: ', '\n')

    for relation in relations_list:
        print(relation)
        G.add_edge(objects[relation.get("subject")], objects[relation.get(
            "object")], label=relation.get("relation"))

    pos = nx.spring_layout(G)
    node_size = 800
    nx.draw(G, with_labels=True, node_size=node_size)
    edge_labels = nx.get_edge_attributes(G, "label")
    label_pos = 0.5
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, label_pos=label_pos)

    plt.show()

    # print(similarity_input)  # the output
    return similarity_input


def spanList(scenegraph):
    entities_list = scenegraph["entities"]
    for entity in entities_list:
        print(entity["lemma_span"])


def scenegraphTable(sentence):
    # Here we just use the default parser.
    parserOutput = sng_parser.parse(sentence)
    print()
    # original output
    print('Default Parser Output: \n', parserOutput)
    sng_parser.tprint(parserOutput)
    return scenegraphGraph(parserOutput), parserOutput


# Scene graph parsing completed here


#########################################################
print("Loading model")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded model")
g = Graph()
g.parse('Populated_Assets_KG.ttl', format="ttl")
n = Namespace("http://www.semanticweb.org/szm/megascan-assets-ontology#")

#######################################################################
# util functions


def create_embedding_tensor(text):
    embedding_tensor = sentence_model.encode(text)
    print()
    return embedding_tensor


def str_to_tensor(string):
    str_list = string.split(',')

    float_list = [float(x) for x in str_list]
    tenser_res = torch.tensor(float_list)
    return tenser_res


def tensor_to_str(tensor):
    tensor_list = tensor.tolist()
    tensor_list_string = ""
    for l in tensor_list:
        tensor_list_string += str(l) + ','
    tensor_list_string = tensor_list_string[:-1]  # remove last ','
    return tensor_list_string


def find_similar_asset(input_string, KG_info):
    cosine_scores = []
    KG_tensor_dict = KG_info[0]
    KG_tensors_values = list(KG_tensor_dict.values())
    for val in KG_tensors_values:
        cosine_scores.append(util.cos_sim(input_string, val))

    # cosine_scores =  util.cos_sim(input_string, list(KG_tensor_dict.values()))
    # Find the pair with the highest cosine similarity score
    max_score = -1
    max_index = -1
    for i in range(len(cosine_scores)):
        if cosine_scores[i] > max_score:
            max_score = cosine_scores[i]
            max_index = i
    closest_asset_ID = ""
    for id, tensor in KG_tensor_dict.items():
        if torch.equal(tensor, KG_tensors_values[max_index]):
            closest_asset_ID = id
            break
    return KG_info[0][closest_asset_ID], KG_info[1][closest_asset_ID], KG_info[2][closest_asset_ID]


#######################################################################
# form embeddings for entities mentioned in input

def scene_graph_tensors(sceneGraph_dict):
    all_spawning_asset_desc = []
    asset_tensor_dict = dict()
    entities = list(sceneGraph_dict.keys())
    for key, value in sceneGraph_dict.items():
        desc = ""
        for modifier in value:
            desc += modifier + ' '
        all_spawning_asset_desc.append(desc+key)

    # list of all tensors for all nouns and their modifiers - later for matching with assets
    all_spawning_asset_tensor = []
    for i in range(len(entities)):
        # for str_desc in all_spawning_asset_desc:
        asset_tensor_desc = create_embedding_tensor(all_spawning_asset_desc[i])
        all_spawning_asset_tensor.append(asset_tensor_desc)
        asset_tensor_dict[entities[i]] = asset_tensor_desc

    return all_spawning_asset_tensor, asset_tensor_dict


def get_KG_asset_tensors():
    qres = g.query(
        """
        PREFIX mega:<http://www.semanticweb.org/szm/megascan-assets-ontology#>
        SELECT ?s ?aID ?aName ?aTensor
        WHERE
        {
            {?s mega:assetTensor ?aTensor}
            UNION
            {?s mega:assetID ?aID}
            UNION
            {?s mega:assetName ?aName}

        }
    """)

    asset_Tensor_dict = {}
    asset_ID_dict = {}
    asset_Name_dict = {}
    count = 0
    for row in qres:
        try:
            asset_Tensor_dict[row.s.toPython()] = str_to_tensor(
                row.aTensor.toPython())
        except Exception as e:
            count += 1
        try:
            asset_ID_dict[row.s.toPython()] = row.aID.toPython()
        except Exception as e:
            count += 1
        try:
            asset_Name_dict[row.s.toPython()] = row.aName.toPython()
        except Exception as e:
            count += 1
    return asset_Tensor_dict, asset_ID_dict, asset_Name_dict


def write_to_file(chosen_asset):
    # print("Chosen Asset ID: ", chosen_asset[1])
    # print("Chosen Asset Name: ", chosen_asset[2])
    # print("\n\n")
    line = re.sub(" ", "_", chosen_asset[2])
    line += "_"+chosen_asset[1]

    # append mode
    fil = open("asset_info.txt", "a")
    fil.write(line)
    fil.write("\n\n\n")
    fil.close()


def match_KG_input(KG_info, input_tensors):
    entity_asset_dict = dict()
    # input_tensors
    # [entity_tensors, entityName_tensor_dictionary]

    for k, v in input_tensors[1].items():
        # for in_tensor in input_tensors[0]:
        # for each input entity, find closest assetID and path+append to file: 'assetName_assetID'
        chosen_asset = find_similar_asset(v, KG_info)
        entity_asset_dict[k] = re.sub(
            " ", "_", chosen_asset[2]) + "_" + chosen_asset[1]
        write_to_file(chosen_asset)
    # print("Entity Asset Dictionary:\n")
    # print(entity_asset_dict)
    return entity_asset_dict


def main():

    sentence = 'There is a small green leaf on a large gray boulder.'
    sentence = 'There is a small green leaf on a large gray boulder. The boulder sits in a wide river bank.'
    sentence = 'There is a small green leaf on a large gray boulder. The boulder with pebbles around it is in a wide river.'

    # Co-reference resolution issue + in,has not detected as relations
    sentence = 'A small green leaf lies on a large gray boulder. The boulder is in a wide river and has small pebbles around it.'

    sentence = 'A small green leaf lies on a large gray boulder. The boulder sits in a wide river. The leaf has small pebbles around it.'
    sentence = 'Cow standing on sidewalk in city area near shops .'

    KG_info = get_KG_asset_tensors()
    # print('Input your own sentence. Type q to quit.')
    # while True:
    #     sentence = input('> ')
    #     if sent.strip() == 'q':
    #         break

    fil = open("asset_info.txt", "w")
    fil.close()
    res_dict, parserOutput = scenegraphTable(sentence)
    res_tensors = scene_graph_tensors(res_dict)
    entity_asset_dict = match_KG_input(KG_info, res_tensors)
    print("Entity Asset Dictionary:\n", entity_asset_dict)
    subject_all_objects_dict = subjectAllObjectsDict(parserOutput)
    print("All Subject-Objects Dictionary:\n", subject_all_objects_dict)
    print("Formatted subject-object dictionary:\n", formatAssetWriting(
        entity_asset_dict, subject_all_objects_dict))


if __name__ == '__main__':
    main()
