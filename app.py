import gensim
from flask import Flask, request, jsonify
import xml.etree.ElementTree as ET
from gensim.models.word2vec import Word2Vec
import numpy as np
import os

from numpy import dot
from numpy.linalg import norm

app = Flask(__name__)


class ArchiComponent:
    archi_components = []
    component_map = {}

    def __init__(self, component_type, component_name, component_id, model_id, model_name, word_vector_avg):
        self.model_id = model_id
        self.component_type = component_type
        self.component_name = component_name
        self.component_id = component_id
        self.model_name = model_name
        self.word_vector_avg = word_vector_avg

    def get_model_id(self):
        return self.model_id

    def get_component_type(self):
        return self.component_type

    def get_component_name(self):
        return self.component_name

    def get_component_id(self):
        return self.component_id

    def get_model_name(self):
        return self.model_name

    def get_word_vector_avg(self):
        return self.word_vector_avg


def get_avg(model, name):
    avg = np.zeros(300)
    if name:
        for word in name.split(" "):
            if word in model:
                avg += model[word]
    return avg


def is_duplicate(archi_components: list, archi_component: ArchiComponent):
    for c in archi_components:
        if c.get_component_name() == archi_component.get_component_name() and c.get_component_type() == archi_component.get_component_type():
            return True
    return False


def load_archi_components():
    archi_components = []
    # Archi models are picked up from -
    # https://github.com/borkdominik/CM2KG/tree/main/Experiments/EMF/Archi/ManyModels/repo-github-archimate/models
    path = 'D://github//archi_models'
    model = gensim.models.KeyedVectors.load_word2vec_format('D:/github/word_2_vec/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    for archi_xml in os.listdir(path):
        file_name = path + "//" + archi_xml
        tree = ET.parse(file_name)
        print(f'Processing....,{file_name}')
        root = tree.getroot()
        model_id = root.attrib['identifier']
        xmlns = root.tag.replace('model', '')
        schema = 'schemaLocation'
        location = ''
        for key in root.keys():
            if schema in key:
                location = key.replace(schema, '')
                break
        model_name = root.find(xmlns + 'name').text
        next_root = root.find(xmlns + 'elements')
        if next_root is not None:
            for child in root.find(xmlns + 'elements'):
                component_id = child.attrib['identifier']
                name = child.find(xmlns + 'name').text
                component_type = child.attrib[location + 'type']
                word_2_vec_avg = get_avg(model, name)
                archi_component = ArchiComponent(component_type, name, component_id, model_id, model_name,
                                                 word_2_vec_avg)
                archi_components.append(archi_component)
    ArchiComponent.archi_components = archi_components
    ArchiComponent.component_map = group_by_type(archi_components)
    return archi_components


def calc_cosine_similarity(component1: ArchiComponent, component2: ArchiComponent):
    v1 = component1.get_word_vector_avg()
    v2 = component2.get_word_vector_avg()
    return dot(v1, v2) / (norm(v1) * norm(v2))


def get_component_names_by_type(component_type: str):
    component_map = ArchiComponent.component_map
    if component_type in component_map:
        component_map = component_map[component_type]
        return list(map(lambda c: [c.get_component_name()], component_map))


def get_component_object(component_id: str):
    for component in ArchiComponent.archi_components:
        if component_id == component.get_component_id():
            return component


def group_by_type(archi_components):
    components_by_type = {}
    a = 0
    for component in archi_components:
        a = a + 1
        component_type = component.get_component_type()
        values = []
        if component_type in components_by_type:
            value = components_by_type[component_type]
            value.append(component)

        else:
            values.append(component)
            components_by_type[component_type] = values
    return components_by_type


# Ensures components are loaded only one time.
def ensure_components_are_loaded():
    if not (bool(ArchiComponent.archi_components) or bool(ArchiComponent.component_map)):
        load_archi_components()


# Response map
def create_json_map(recommendations: dict):
    response_list = []
    for recommendation, score in recommendations:
        component_info_map = {}
        component_map = {}
        component_info_map['id'] = recommendation.get_component_id()
        component_info_map['name'] = recommendation.get_component_name()
        component_info_map['type'] = recommendation.get_component_type()
        component_info_map['modelId'] = recommendation.get_model_id()
        component_map['component'] = component_info_map
        component_map['score'] = score
        response_list.append(component_map)
    return response_list


def get_components_from_name(component_type: str, similar_components):
    component_map = {}
    for component, score in similar_components:
        for archi_component in ArchiComponent.archi_components:
            if component == archi_component.get_component_name() and archi_component.get_component_type() == component_type:
                component_map[archi_component] = score
                break
    return component_map


# Model that calculates similairty and provides recommendations
def find_similar_components_word2vec(component_object: ArchiComponent):
    component_names = list(
        map(lambda c: [c.get_component_name()], ArchiComponent.component_map[component_object.get_component_type()]))
    model = Word2Vec(sentences=component_names, vector_size=64, sg=1, window=1, min_count=2, workers=8)
    similar_components = model.wv.most_similar(component_object.get_component_name())
    return similar_components


def find_similar_components(component: ArchiComponent):
    filtered_components = filter(lambda c: c.get_model_id() != component.get_model_id(),
                                 ArchiComponent.component_map[component.get_component_type()])
    score_map = {}
    for filtered_component in filtered_components:
        score_map[filtered_component] = calc_cosine_similarity(filtered_component, component)
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    ensure_components_are_loaded()
    component_id = request.args.get('id')
    component_object = get_component_object("id-"+component_id)
    if component_object:
        similar_components = find_similar_components(component_object)
        return jsonify(create_json_map(list(similar_components)[:10]))
    return jsonify([])


if __name__ == '__main__':
    app.run(debug=True)
