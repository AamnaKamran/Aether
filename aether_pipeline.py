# from wrmd_sim.wrmd_similarity import find_dist
# import speech_recognition as sr

# flag = 1

# while (flag):

#     r = sr.Recognizer()

#     with sr.Microphone() as source:
#         r.adjust_for_ambient_noise(source)

#         print("Please say something")

#         audio = r.listen(source)

#         print("Recognizing Now .... ")

#        # recognize speech using google

#         try:
#             print("\nYou have said: \n" + r.recognize_google(audio) + "\n")
#             user_input = input("Press\n- 0, if this is incorrect:\n- 1, if this is correct ")
#             if int(user_input) == 0:
#                 flag = 1
#             elif int(user_input) == 1:
#                     flag = 0

#         except Exception as e:
#             print("Error :  " + str(e))

#     # write audio
#     with open("recorded.wav", "wb") as f:
#         f.write(audio.get_wav_data())


# import unreal
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

# populated_kg_path = "/home/aim2/Documents/Populated_Assets_KG.ttl"
populated_kg_path = (
    "Populated_Assets_KG.ttl"
)
# Update KG of assets
#################################################
import os
import json

# find if directory is '../Megascans Library/Downloaded/UAssets' or '../Megascans Library/UAssets'
path_to_files = "/home/aim2/Documents/DriveA/UAssets/"
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import FOAF, XMLNS, XSD, RDF, RDFS, OWL
import os

g = Graph()
g.parse(populated_kg_path, format="ttl")
n = Namespace("http://www.semanticweb.org/szm/megascan-assets-ontology#")


# find if directory is '../Megascans Library/Downloaded/UAssets' or '../Megascans Library/UAssets'
def trigger_update(folder_path):
    qres = g.query(
        """
      PREFIX mega:<http://www.semanticweb.org/szm/megascan-assets-ontology#>
      SELECT ?s ?aID ?aName WHERE {
      ?s mega:assetID ?aID.
      ?s mega:assetName ?aName.
       FILTER BOUND(?aID)
    }"""
    )

    cur_KG_assets = set()
    for row in qres:
        # check asset IDs currently loaded in the assets' KG - they have IDs same as their folder and json file names
        cur_KG_assets.add(row.aID.toPython())
    cur_KG_assets = list(cur_KG_assets)
    saved_folders = os.listdir(folder_path)
    saved_asset_paths = dict()
    # sf = asset folder name
    # sf - also asset ID
    for sf in saved_folders:
        try:
            if len(os.listdir(str(folder_path) + str(sf))) != 0 and str(
                sf
            ) + ".json" in os.listdir(str(folder_path) + str(sf) + "/"):
                saved_asset_paths[sf] = (
                    str(folder_path) + str(sf) + "/" + str(sf) + ".json"
                )
        except Exception as e:
            if str(e).find("NotADirectoryError") != -1:
                print("Actual error:", e)

    print("Paths for assets in file system: ", saved_asset_paths)

    new_paths_dict = dict()
    for key, value in saved_asset_paths.items():
        if key not in cur_KG_assets:
            new_paths_dict[key] = value
    print("Assets not yet added to KG:", new_paths_dict)
    return new_paths_dict


def update_KG(new_files_paths):
    print("New files path: ", new_files_paths, " ", len(new_files_paths))
    # correct

    n = Namespace("http://www.semanticweb.org/szm/megascan-assets-ontology#")
    # json_files = [pos_json for pos_json in os.listdir(
    #     new_files_paths) if pos_json.endswith('.json')]
    json_files = list(new_files_paths.values())

    for fil in json_files:
        # Open asset file
        try:
            f = open(fil)
            data = json.load(f)  # json file loaded as dictionary

            # Create asset and set asset name
            # obtain name of asset, convert it to RDF format literal
            name = data["name"]
            name_processed = re.sub("[^a-zA-Z0-9]", "_", name)
            # print("Created asset name = ", name)
            print(name_processed)
            # create a resource in the Assets Knowledge Graph (KG)
            asset = URIRef(str(n) + "Asset_" + name_processed)
            # print("Asset = ", asset)
            g.add((asset, RDF.type, n.Asset))
            # Add triple to KG: {new Asset resource, nameRelation (from schema), name obtained from asset file}
            g.add((asset, n.assetName, Literal(name, datatype=XSD.string)))

            try:
                # Set asset ID
                id = Literal(str(data["id"]), datatype=XSD.string)
                g.add((asset, n.assetID, id))
                print("Added assetID = ", id)
            except:
                print("No asset ID specified for asset", asset)

            try:
                # Set asset search tags
                tags = data["tags"]
                # print("number of tags = ", len(tags))
                for t in tags:
                    tag = Literal(str(t), datatype=XSD.string)
                    g.add((asset, n.assetTag, tag))
                    # print("Added tag = ", tag)

            except:
                print("No search tags specified for asset", asset)

            try:
                # Create category individuals and set their names, then assign asset categories
                categories = data["categories"]
                # print("number of categories = ", len(categories))
                for c in categories:
                    category = Literal(c, datatype=XSD.string)
                    g.add((asset, n.assetCategory, category))
                    # print("Adding category = ", category)

            except:
                print("No categories specified for asset", asset)

            properties = data["properties"]
            # print("number of properties = ", len(properties))
            for prop in properties:
                key = list(prop.keys())[0]
                value = list(prop.values())[0]
                # print("prop['key'] = ", prop["key"])
                # print("prop['value'] = ", prop["value"])

            try:
                properties = data["properties"]
                # print("number of properties = ", len(properties))
                for prop in properties:
                    key = list(prop.keys())[0]
                    value = list(prop.values())[0]

                    # print("prop['key'] = ", prop["key"])
                    # print("prop['value'] = ", prop["value"])

                    # print("prop key = ", key)
                    if prop["key"] == "size":
                        # print(value, Literal(value))
                        if prop["value"] == "tiny":
                            g.add((asset, n.assetSize, n.tiny))
                            # print("Added asset size = ", n.tiny)
                        elif prop["value"] == "small":
                            g.add((asset, n.assetSize, n.small))
                            # print("Added asset size = ", n.small)
                        elif prop["value"] == "medium":
                            g.add((asset, n.assetSize, n.medium))
                            # print("Added asset size = ", n.medium)
                        elif prop["value"] == "large":
                            g.add((asset, n.assetSize, n.large))
                            # print("Added asset size = ", n.large)
                        else:
                            g.add((asset, n.assetSize, n.extra_large))
                            # print("Added asset size = ", n.extra_large)

                    elif prop["key"] == "age":
                        age = Literal(prop["value"], datatype=XSD.string)
                        g.add((asset, n.assetAge, age))
                        # print("Added asset age = ", age)

            except:
                print("No properties specified for asset", asset)

            try:
                # Set asset average color
                avg_color = Literal(str(data["averageColor"]), datatype=XSD.string)
                g.add((asset, n.assetAvgColor, avg_color))
                # print("Added asset average color = ", avg_color)
            except:
                print("No asset avg color specified for asset", asset)
            print(len(g))
            f.close()
        except Exception as e:
            print("Update error: ", e)
        g.serialize(destination="Populated_Assets_KG.ttl", format="turtle")


#################################################
def find_dist_cosine(sen_a, sen_b):
    return util.cos_sim(sen_a, sen_b)


def sub_closest_prep(preposition):
    # above
    prep_list = [
        "on",
        "back",
        "front",
        "right",
        "left",
        "behind",
        "beneath",
        "in",
        "over",
        "under",
        "besides",
    ]
    min_dist = -1000
    target_prep = ""
    for candidate in prep_list:
        dist = find_dist_cosine(
            create_embedding_tensor(candidate), create_embedding_tensor(preposition)
        )
        if dist > min_dist:
            min_dist = dist
            target_prep = "" + candidate
    print("\nClosest preposition for ", preposition, " is ", target_prep)
    return target_prep


# def find_dist(sen_a, sen_b, model_name):

#     loaded_model = FastText.load(model_name)
#     wv = loaded_model.wv
#     distance = wv.wmdistance(sen_a, sen_b)
#     return distance

N_id = 1


jar_path = "stanford-corenlp-4.5.1/stanford-corenlp-4.5.1.jar"
models_jar_path = "stanford-corenlp-4.5.1/stanford-corenlp-4.5.1-models.jar"


def formatAssetWriting(entity_asset_tuple, subject_all_objects_dict):
    formattedDict = dict()

    for k, v in subject_all_objects_dict.items():
        formattedKey = k
        formattedVal = v
        if k[: k.rfind("_")] == entity_asset_tuple[0]:
            print(
                "entity_asset_tuple:",
                entity_asset_tuple[2].replace(" ", "_")
                + "_"
                + entity_asset_tuple[1]
                + k[k.rfind("_") :],
            )

            formattedKey = (
                entity_asset_tuple[2].replace(" ", "_")
                + "_"
                + entity_asset_tuple[1]
                + k[k.rfind("_") :]
            )

        for values in v:
            prep, val = list(values)
            if val[: val.rfind("_")] == entity_asset_tuple[0]:
                print(
                    "preposiiton - entity_asset_tuple:",
                    prep,
                    " ",
                    entity_asset_tuple[2].replace(" ", "_")
                    + "_"
                    + entity_asset_tuple[1]
                    + val[val.rfind("_") :],
                )
                formattedVal = (
                    prep,
                    entity_asset_tuple[2].replace(" ", "_")
                    + "_"
                    + entity_asset_tuple[1]
                    + val[val.rfind("_") :],
                )
                print("Formatted prep-values = ", prep + " " + val)
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

    print("Obtained scene graph:", scenegraph)

    objects = []
    entities_list = scenegraph["entities"]
    print("Entities List: ", entities_list)
    for entity in entities_list:
        objects.append(entity["head"])

    # if :

    relations_list = scenegraph["relations"]
    print("relations", relations_list)
    max_ID = 0
    print("Printing relations: ", "\n")
    subject_objects_dict = dict()
    for relation in relations_list:
        # subject+'_'+uniqueEntityID
        subject = objects[relation.get("subject")] + "_" + str(relation["subject"])
        if relation["subject"] > max_ID:
            max_ID = relation["subject"]

        relation_object = (
            relation["relation"],
            objects[relation.get("object")] + "_" + str(relation["object"]),
        )
        if relation["object"] > max_ID:
            max_ID = relation["object"]

        if subject in list(subject_objects_dict.keys()):
            subject_objects_dict[subject].append(relation_object)
        else:
            subject_objects_dict[subject] = [relation_object]
    #            print("Setting key-value ", subject,
    #                  ": ", subject_objects_dict[subject])

    for entity in entities_list:
        if entity["head"] not in list(subject_objects_dict.keys()) and entity[
            "head"
        ] not in list(subject_objects_dict.values()):
            max_ID += 1
            #            print("No relation found.")
            subject_objects_dict[entity["head"] + "_" + str(max_ID)] = [("", "")]

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
            similarity_input[object].append(modifier)

    # RELATIONS
    relations_list = scenegraph["relations"]
    print("Printing relations: ", "\n")

    for relation in relations_list:
        print(relation)
        G.add_edge(
            objects[relation.get("subject")],
            objects[relation.get("object")],
            label=relation.get("relation"),
        )

    pos = nx.spring_layout(G)
    node_size = 800
    nx.draw(G, with_labels=True, node_size=node_size)
    edge_labels = nx.get_edge_attributes(G, "label")
    label_pos = 0.5
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=label_pos)

    plt.title("Scene Graph of Input")
    plt.show()
    return similarity_input


def spanList(scenegraph):
    entities_list = scenegraph["entities"]
    for entity in entities_list:
        print(entity["lemma_span"])


def scenegraphTable(sentence):
    # Here we just use the default parser.
    parserOutput = sng_parser.parse(sentence)
    print("Default Parser Output: \n", parserOutput)
    sng_parser.tprint(parserOutput)
    return scenegraphGraph(parserOutput), parserOutput


# Scene graph parsing completed here


#########################################################
print("Loading SentenceTransformer Model...")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loaded.")
g = Graph()
g.parse(populated_kg_path, format="ttl")
n = Namespace("http://www.semanticweb.org/szm/megascan-assets-ontology#")

#######################################################################
# util functions


def create_embedding_tensor(text):
    embedding_tensor = sentence_model.encode(text)
    # print()
    return embedding_tensor


def str_to_tensor(string):
    str_list = string.split(",")

    float_list = [float(x) for x in str_list]
    tenser_res = torch.tensor(float_list)
    return tenser_res


def tensor_to_str(tensor):
    tensor_list = tensor.tolist()
    tensor_list_string = ""
    for l in tensor_list:
        tensor_list_string += str(l) + ","
    tensor_list_string = tensor_list_string[:-1]  # remove last ','
    return tensor_list_string


def concatAssetIntoSen(KG_info):
    assetSentenceForm = dict()
    try:
        for k, v in list(KG_info[1].items()):  # dict of asset ID and Name
            # print("setting for asset id:", k.replace(k[:k.rfind("_"), ""]))
            assetSentenceForm[k[k.rfind("_") + 1 :]] = "" + v
    except Exception as e:
        print("Name adding unsuccessful", e)

    try:
        for k, v in list(KG_info[2].items()):  # dict of asset Tag
            assetSentenceForm[k[k.rfind("_") + 1 :]] += " " + v
    except:
        print("Tag adding unsuccessful")

    try:
        # dict of asset Size - placed size at start
        for k, v in list(KG_info[3].items()):
            assetSentenceForm[k[k.rfind("_") + 1 :]] = (
                v[v.rfind("#") + 1 :]
                + " sized "
                + assetSentenceForm[k[k.rfind("_") + 1 :]]
            )
    except:
        print("Size adding unsuccessful")
    return assetSentenceForm


def find_similar_asset_cos(in_asset_desc_pair, KG_info):
    wrmd_scores = []
    in_asset_Name = list(in_asset_desc_pair.keys())[0]
    in_asset_Desc = list(in_asset_desc_pair.values())[0]
    # KG_info -> [assetID dict, asset Names dict, asset Tags dict, asset Sizes dict]
    KG_info_dict = concatAssetIntoSen(KG_info)
    print()

    KG_sentences_vals = list(KG_info_dict.values())
    for val in KG_sentences_vals:
        wrmd_scores.append(
            find_dist_cosine(
                create_embedding_tensor(in_asset_Desc), create_embedding_tensor(val)
            )
        )
    max_score = -1
    max_index = -1
    for i in range(len(wrmd_scores)):
        if wrmd_scores[i] > max_score:
            max_score = wrmd_scores[i]
            max_index = i
    closest_asset_ID = ""
    for id, sen in KG_info_dict.items():
        if sen == KG_sentences_vals[max_index]:
            closest_asset_ID = id
            break
    try:
        KG_info[0][
            str("http://www.semanticweb.org/szm/megascan-assets-ontology#Asset_")
            + closest_asset_ID
        ]
    except Exception as e:
        print("e for asset ID: ", e)
    try:
        KG_info[1][
            str("http://www.semanticweb.org/szm/megascan-assets-ontology#Asset_")
            + closest_asset_ID
        ]
    except Exception as e:
        print("e for asset Name: ", e)
    try:
        KG_info[2][
            str("http://www.semanticweb.org/szm/megascan-assets-ontology#Asset_")
            + closest_asset_ID
        ]
    except Exception as e:
        print("e for asset Tags: ", e)
    try:
        KG_info[3][
            str("http://www.semanticweb.org/szm/megascan-assets-ontology#Asset_")
            + closest_asset_ID
        ]
    except Exception as e:
        print("e for asset Size: ", e)

    # now return the entity name against which the match was being asset ID and Name for file writing formatting
    return (
        in_asset_Name,
        KG_info[0][
            str("http://www.semanticweb.org/szm/megascan-assets-ontology#Asset_")
            + closest_asset_ID
        ],
        KG_info[1][
            str("http://www.semanticweb.org/szm/megascan-assets-ontology#Asset_")
            + closest_asset_ID
        ],
    )


def find_similar_asset(input_string, KG_info):
    cosine_scores = []
    KG_tensor_dict = KG_info[0]
    KG_tensors_values = list(KG_tensor_dict.values())
    for val in KG_tensors_values:
        cosine_scores.append(util.cos_sim(input_string, val))
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
    return (
        KG_info[0][closest_asset_ID],
        KG_info[1][closest_asset_ID],
        KG_info[2][closest_asset_ID],
    )


#######################################################################
# form embeddings for entities mentioned in input


def scene_graph_list(sceneGraph_dict):
    print("\nSCENE GRAPH DICT\n:", sceneGraph_dict)
    all_spawning_asset_desc = []
    asset_list_dict = dict()
    entities = list(sceneGraph_dict.keys())
    for key, value in sceneGraph_dict.items():
        desc = ""
        for modifier in value:
            desc += modifier + " "
        all_spawning_asset_desc.append(desc + key)

    # list of all tensors for all nouns and their modifiers - later for matching with assets
    all_spawning_asset_list = []
    for i in range(len(entities)):
        # for str_desc in all_spawning_asset_desc:
        asset_desc = all_spawning_asset_desc[i]
        all_spawning_asset_list.append(asset_desc)
        asset_list_dict[entities[i]] = asset_desc

    return all_spawning_asset_list, asset_list_dict


def scene_graph_tensors(sceneGraph_dict):
    all_spawning_asset_desc = []
    asset_tensor_dict = dict()
    entities = list(sceneGraph_dict.keys())
    for key, value in sceneGraph_dict.items():
        desc = ""
        for modifier in value:
            desc += modifier + " "
        all_spawning_asset_desc.append(desc + key)

    # list of all tensors for all nouns and their modifiers - later for matching with assets
    all_spawning_asset_tensor = []
    for i in range(len(entities)):
        # for str_desc in all_spawning_asset_desc:
        asset_tensor_desc = create_embedding_tensor(all_spawning_asset_desc[i])
        all_spawning_asset_tensor.append(asset_tensor_desc)
        asset_tensor_dict[entities[i]] = asset_tensor_desc

    return all_spawning_asset_tensor, asset_tensor_dict


def get_asset_concatTags_dict(tags_list):
    asset_concatTags_dict = {}
    # iterate over list - for all elements[0] being same, concatenate to tags_by_asset_xyz to form a sentence
    for elem in tags_list:
        if elem[0] in asset_concatTags_dict.keys():
            asset_concatTags_dict[elem[0]] += str(" " + asset_concatTags_dict[elem[1]])
        else:
            asset_concatTags_dict[elem[0]] = str(asset_concatTags_dict[elem[1]])
    print("Finally, assets with all tags in sentence:\n", asset_concatTags_dict)
    return asset_concatTags_dict


def get_KG_assets():
    qres = g.query(
        """
        PREFIX mega:<http://www.semanticweb.org/szm/megascan-assets-ontology#>
        SELECT ?s ?aID ?aName ?aTag ?aSize
        WHERE
        {
            {?s mega:assetTag ?aTag}
            UNION
            {?s mega:assetID ?aID}
            UNION
            {?s mega:assetName ?aName}
            UNION
            {?s mega:assetSize ?aSize}
        }
    """
    )

    asset_Tag_dict = {}
    asset_ID_dict = {}
    asset_Name_dict = {}
    asset_Size_dict = {}
    count = 0
    for row in qres:
        try:
            asset_ID_dict[row.s.toPython()] = row.aID.toPython()
        except Exception as e:
            count += 1
        try:
            asset_Name_dict[row.s.toPython()] = row.aName.toPython()
        except Exception as e:
            count += 1
        try:
            asset_Size_dict[row.s.toPython()] = row.aSize.toPython()
        except Exception as e:
            count += 1
        try:
            if row.aTag is not None:
                # print("Adding tags for ", row.s.toPython())
                asset_Tag_dict[row.s.toPython()] += " " + row.aTag.toPython()

            # print("Updated asset tags list")
        except Exception as e:
            if row.aTag is not None:
                asset_Tag_dict[row.s.toPython()] = row.aTag.toPython()
            count += 1

    return asset_ID_dict, asset_Name_dict, asset_Tag_dict, asset_Size_dict


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
    """
    )

    asset_Tensor_dict = {}
    asset_ID_dict = {}
    asset_Name_dict = {}
    count = 0
    for row in qres:
        try:
            asset_Tensor_dict[row.s.toPython()] = str_to_tensor(row.aTensor.toPython())
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


def entity_sub_id(entity, all_assets_dicts):
    #    print("entity = ", entity)
    #    print("all_assets_dict = ", all_assets_dicts)
    if type(entity) == "tuple":
        entity1 = entity[1]
    else:
        entity1 = entity
    if entity1[: entity1.rfind("_")] in list(all_assets_dicts.keys()):
        return (
            all_assets_dicts[entity1[: entity1.rfind("_")]]
            + entity1[entity1.rfind("_") :]
        )


def write_all_prep_to_file(all_chosen_assets, subj_obj_dict):
    #    print("subj_obj_dict = ", subj_obj_dict)
    print("all_chosen_assets = ", all_chosen_assets)

    formattedDict = dict()
    for k, v in subj_obj_dict.items():
        if len(v[0][0]) != 0 and len(v[0][1]) != 0:
            print("k,v pair = ", k, "\t", v, "\n")
            if entity_sub_id(k, all_chosen_assets) in list(formattedDict.keys()):
                for elem in v:
                    #                    print("Inner tuple: ", elem)
                    formattedDict[entity_sub_id(k, all_chosen_assets)].append(
                        (elem[0], entity_sub_id(elem[1], all_chosen_assets))
                    )
            else:
                for elem in v:
                    #                    print("Inner tuple: ", elem)
                    # if elem[0]
                    formattedDict[entity_sub_id(k, all_chosen_assets)] = [
                        (elem[0], entity_sub_id(elem[1], all_chosen_assets))
                    ]

        else:
            formattedDict[entity_sub_id(k, all_chosen_assets)] = [("", "")]

    #    print("final Formatted Dict = ", formattedDict)
    return formattedDict


def write_to_file(chosen_asset, subj_obj_dict):
    # write_prep_to_file()
    formatted_writing = formatAssetWriting(chosen_asset, subj_obj_dict)

    line = re.sub(" ", "_", chosen_asset[2])
    line += "_" + chosen_asset[1]

    # append mode
    fil = open("/home/aim2/Documents/asset_info.txt", "a")
    fil.write(line)
    fil.write("\n\n\n")
    fil.close()


def match_KG_input_cos(KG_info, desc_dict, subj_obj_dict):
    # asset_Tag_dict, asset_Size_dict, asset_ID_dict, asset_Name_dict

    entity_asset_dict = dict()
    # input_tensors
    # [entity_tensors, entityName_tensor_dictionary]
    # print("Input Desc:", desc_dict)
    all_chosen_assets = dict()
    for k, v in desc_dict.items():
        # for in_tensor in input_tensors[0]:
        # for each input entity, find closest assetID and path+append to file: 'assetName_assetID'
        chosen_asset = find_similar_asset_cos({k: v}, KG_info)
        #        print("Chosen asset = ", chosen_asset)
        all_chosen_assets[chosen_asset[0]] = (
            chosen_asset[2].replace(" ", "_") + "_" + chosen_asset[1]
        )
        entity_asset_dict[k] = re.sub(" ", "_", chosen_asset[2]) + "_" + chosen_asset[1]

    return entity_asset_dict, write_all_prep_to_file(all_chosen_assets, subj_obj_dict)


def match_KG_input(KG_info, input_tensors):
    entity_asset_dict = dict()
    for k, v in input_tensors[1].items():
        chosen_asset = find_similar_asset(v, KG_info)
        entity_asset_dict[k] = re.sub(" ", "_", chosen_asset[1]) + "_" + chosen_asset[0]
        write_to_file(chosen_asset)

    return entity_asset_dict


# def spawn_object_behind(first_actor, second_actor):
#    print("In BEHIND Function")
#    camera_location, camera_rotation = unreal.EditorLevelLibrary.get_level_viewport_camera_info()
#    #camera_rotation.get_forward_vector()
#
#    #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_width = first_actor_bounding_box.box_extent.x
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#
#    second_actor_width = second_actor_bounding_box.box_extent.x
#    vec = unreal.Vector(first_actor_width+ second_actor_width + 200.0, 0, 0)
#
#    second_actor.set_actor_location(first_actor_loc - vec, False, True)
#
# def spawn_object_at_back_of(first_actor, second_actor):
#    print("In BACK OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_width = first_actor_bounding_box.box_extent.x
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_width = second_actor_bounding_box.box_extent.x
#    vec = unreal.Vector(first_actor_width+ second_actor_width, 0, 0)
#
#    second_actor.set_actor_location(first_actor_loc - vec, False, True)
#
# def spawn_object_in_front_of(first_actor, second_actor):
#    print("In FRONT OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_width = first_actor_bounding_box.box_extent.x
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_width = second_actor_bounding_box.box_extent.x
#    vec = unreal.Vector(first_actor_width+ second_actor_width, 0, 0)
#
#    second_actor.set_actor_location(first_actor_loc + vec, False, True)
#
#
# def spawn_object_at_right_of(first_actor, second_actor):
#    print("In RIGHT OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_length = first_actor_bounding_box.box_extent.y
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_length = second_actor_bounding_box.box_extent.y
#    vec = unreal.Vector(0,first_actor_length+ second_actor_length, 0)
#
#    second_actor.set_actor_location(first_actor_loc + vec, False, True)
#
# def spawn_object_at_left_of(first_actor, second_actor):
#    print("In LEFT OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_length = first_actor_bounding_box.box_extent.y
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_length = second_actor_bounding_box.box_extent.y
#    vec = unreal.Vector(0,first_actor_length+ second_actor_length, 0)
#
#    second_actor.set_actor_location(first_actor_loc - vec, False, True)
#
#
# def spawn_object_on(first_actor, second_actor):
#    print("In ON OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_height = first_actor_bounding_box.box_extent.z
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_height = second_actor_bounding_box.box_extent.z
#    vec = unreal.Vector( 0, 0,first_actor_height + second_actor_height)
#
#    second_actor.set_actor_location(first_actor_loc + vec, False, True)
#
# def spawn_object_beneath(first_actor, second_actor):
#    print("In BENEATH OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_height = first_actor_bounding_box.box_extent.z
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_height = second_actor_bounding_box.box_extent.z
#    vec = unreal.Vector( 0, 0,first_actor_height + second_actor_height)
#
#    first_actor.set_actor_location(second_actor_loc + vec, False, True)
#
# def spawn_object_in(first_actor, second_actor):
#    print("In IN OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_height = first_actor_bounding_box.box_extent.z
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_height = second_actor_bounding_box.box_extent.z
#    #vec = unreal.Vector( 0, 0,first_actor_height + second_actor_height)
#
#    #first_actor.set_actor_location(second_actor_loc + vec, False, True)
#
# def spawn_object_over(first_actor, second_actor):
#    print("In OVER OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_height = first_actor_bounding_box.box_extent.z
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_height = second_actor_bounding_box.box_extent.z
#    vec = unreal.Vector( 0, 0,first_actor_height + second_actor_height + 250)
#
#    second_actor.set_actor_location(first_actor_loc + vec, False, True)
#
# def spawn_object_under(first_actor, second_actor):
#    print("In UNDER OF Function")
#     #Fisrt Actor
#    first_actor_loc = first_actor.get_actor_location()
#    first_actor_static_mesh_component = first_actor.get_component_by_class(unreal.StaticMeshComponent)
#    first_actor_static_mesh = first_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    first_actor_bounding_box = first_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    first_actor_height = first_actor_bounding_box.box_extent.z
#    #Second Actor
#    second_actor_loc = second_actor.get_actor_location()
#    second_actor_static_mesh_component = second_actor.get_component_by_class(unreal.StaticMeshComponent)
#    second_actor_static_mesh = second_actor_static_mesh_component.static_mesh
#
#    # Get the bounding box of the static mesh component
#    second_actor_bounding_box = second_actor_static_mesh.get_bounds()
#
#    # Extract the height from the bounding box
#    second_actor_height = second_actor_bounding_box.box_extent.z
#    vec = unreal.Vector( 0, 0,first_actor_height + second_actor_height+ 250)
#
#    first_actor.set_actor_location(second_actor_loc + vec, False, True)
#
# def spawn_object_besides(first_actor, second_actor):
#    spawn_object_at_right_of(first_actor, second_actor)
#
# def spawn_assets(asset_dicts):
#    asset_directory = "/Game/Megascans/3D_Assets/"
#    file1 = open('/home/aim2/Documents/asset_info.txt', 'r')
#    Lines = file1.readlines()
#    #class_path = "/Script/Project1.CustomizableActor"
#    for sub,objs in asset_dicts.items():
#        print(f"Asset = {sub}")
#        prev_actor = spawn_an_asset(sub)
#        if prev_actor != None:
#            for obj in objs:
#                if obj[0] != "" and obj[1] != "":
#                    print(f"Obj1 = {obj[1]}")
#                    actor = spawn_an_asset(obj[1])
#                    if actor!= None:
#                        if obj[0] == "on":
#                            spawn_object_on(prev_actor, actor)
#                        elif obj[0] == "back":
#                            spawn_object_at_back_of(prev_actor, actor)
#                        elif obj[0] == "front":
#                            spawn_object_in_front_of(prev_actor, actor)
#                        elif obj[0] == "right":
#                            spawn_object_at_right_of(prev_actor, actor)
#                        elif obj[0] == "left":
#                            spawn_object_at_left_of(prev_actor, actor)
#                        elif obj[0] == "behind":
#                            spawn_object_behind(prev_actor, actor)
#                        elif obj[0] == "beneath":
#                            spawn_object_beneath(prev_actor, actor)
#                        elif obj[0] == "in":
#                            spawn_object_in(prev_actor, actor)
#                        elif obj[0] == "over":
#                            spawn_object_over(prev_actor, actor)
#                        elif obj[0] == "under":
#                            spawn_object_under(prev_actor, actor)
#                        elif obj[0] == "besides":
#                            spawn_object_besides(prev_actor, actor)
#        else:
#            print("first actor didnt spawn")
#
# def spawn_an_asset(asset_name):
#    asset_directory = "/Game/Megascans/3D_Assets/"
#    world = unreal.EditorLevelLibrary.get_editor_world()
#
#    # Spawn an actor
#    actor_location = unreal.Vector(0.0, 0.0, 0.0)
#    actor_rotation = unreal.Rotator(0.0, 0.0, 0.0)
#
#    actor_class = unreal.EditorAssetLibrary.load_blueprint_class("/Game/Blueprints/MyActor")
#    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, actor_location, actor_rotation)
#
#
#    asset_registry_module = unreal.AssetRegistryHelpers.get_asset_registry()
#    assets = asset_registry_module.get_assets_by_path(asset_directory , recursive=True, include_only_on_disk_assets=False)
#
#
#    filter = unreal.ARFilter(recursive_paths=True)
#    filter.class_names.append("StaticMesh")
#    assets = asset_registry_module.run_assets_through_filter(assets, filter)
#    #print(f"assets = {assets}")
#
#    f_asset_name = asset_name[:asset_name.rfind('_')]
#    for asset in assets:
#
#        asset_str = str(asset.asset_name)
#        asset_str = asset_str.strip()
#        line_str = "S_" + f_asset_name.strip()
#        line_str = line_str.strip()
#
#        if asset_str.find(line_str) != -1:
#
#            print("Matched ", asset_str, " to ", line_str)
#            print(asset)
#            #actor.static_mesh_component.set_static_mesh(static_mesh)
#            static_mesh = unreal.EditorAssetLibrary.load_asset(str(asset.object_path))
#            static_mesh_component = actor.get_component_by_class(unreal.StaticMeshComponent)
#
#            # Set the static mesh of the component
#            static_mesh_component.set_static_mesh(static_mesh)
#
#            editor_world = unreal.EditorLevelLibrary.get_editor_world()
#            if editor_world:
#                print("Hello")
#                camera_location, camera_rotation = unreal.EditorLevelLibrary.get_level_viewport_camera_info()
#
#                # Calculate the spawn location in front of the camera
#                spawn_distance = 500.0  # How far in front of the camera to spawn the asset
#                print(f"forward vector{camera_rotation}")
#                spawn_location = camera_location + (camera_rotation.get_forward_vector() * spawn_distance)
#                spawn_location.z = 0.0  # Set the Z coordinate to ground level
#                actor.set_actor_location(spawn_location, False, True)
#                return actor
#
#    return None
#


def main():
    # Co-reference resolution issue + in,has not detected as relations
    #    sentence = 'A small green leaf lies on a large gray boulder. The boulder is in a wide river and has small pebbles around it.'

    # KG_info = get_KG_asset_tensors()
    # new_file_paths = trigger_update(path_to_files)
    # update_KG(new_file_paths)

    sentence = "A tree next to the large boulder"
    KG_assets_only_info = get_KG_assets()

    # print('Input your own sentence. Type q to quit.')
    # while True:
    #     sentence = input('> ')
    #     if sent.strip() == 'q':
    #         break

    #    fil = open("/home/aim2/Documents/asset_info.txt", "w")
    #    fil.close()
    res_dict, parserOutput = scenegraphTable(sentence)
    res_dict = scene_graph_list(res_dict)[1]
    subject_all_objects_dict = subjectAllObjectsDict(parserOutput)
    #    print("subject_all_objects_dict = ", subject_all_objects_dict)
    entity_asset_dict, spawning_dict = match_KG_input_cos(
        KG_assets_only_info, res_dict, subject_all_objects_dict
    )

    #    print("Entity Asset Dictionary:\n", entity_asset_dict)

    final_spawning_dict = dict()
    for k, v in list(spawning_dict.items()):
        final_spawning_dict[k] = []
        for tuple in v:
            if len(v[0][0]) != 0 and len(v[0][1]) != 0:
                final_spawning_dict[k].append((sub_closest_prep(tuple[0]), tuple[1]))
            else:
                final_spawning_dict[k].append(("", ""))

    print("Final Spawning Dict Format: \n", final_spawning_dict)


#    spawn_assets(final_spawning_dict)


if __name__ == "__main__":
    main()
