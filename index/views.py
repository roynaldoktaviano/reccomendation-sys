from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from bs4 import BeautifulSoup
from django.urls import reverse
import requests
import pandas as pd
import numpy as np
import csv
import os
# Importing the libraries
from urllib.request import urlopen
from time import time
from django.views.decorators.csrf import csrf_exempt


status_desc = 0
has_og = 0
has_title = 0
has_alt = 0
load_status = 0
load_num = 0


@csrf_exempt
def indexs(request):
    data = request.POST.get('url_input')
    url_name = data
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


@csrf_exempt
def diagnose(request):

    # # Get input value from url input
    data = request.POST.get('url')

    # Turn url text to lxml
    url_text = requests.get(data).text
    soup = BeautifulSoup(url_text, 'lxml')

    # Get Meta Title
    title = soup.find('title').text
    if title != None:
        has_title = 1
    else:
        has_title = 0

    # Get Description
    desc = soup.find('meta', attrs={'name': 'description'})
    if desc is None:
        status_desc = 0
    else:
        desc_con = soup.find(
            'meta', attrs={'name': 'd   escription'}).get('content')
        jumlah_desc = len(desc_con)
        # Cek panjang karakter meta description
        # Jika dibawah 120 maka terlalu pendek, jika diataas 156 terlalu panjang
        if jumlah_desc < 120:
            status_desc = 1
        elif jumlah_desc > 156:
            status_desc = 2
        elif jumlah_desc == 0:
            status_desc = 0
        elif jumlah_desc > 120 and jumlah_desc < 156:
            status_desc = 3

    # Cek Image and alt
    images = soup.find_all('img', alt=True)
    noalt = []
    i = 0

    while i < len(images):
        if images[i]['alt'] == "":
            noalt.append(images[i]['src'])

        i += 1

    if noalt == []:
        has_alt = 1
    else:
        has_alt = 0

    # Cek OG Image
    og_image = soup.find('meta', property='og:image')
    if og_image is None:
        has_og = 0
    else:
        has_og = 1

    # Get H1 from website
    h1 = soup.find('h1')
    if h1 is None:
        tag_h1 = 0
    else:
        tag_h1 = 1

    # Python program to check the
    # loading time of the website

    # Obtaining the URL of website
    website = urlopen(data)
    # Return the number of seconds
    # passed since epoch
    open_time = time()

    # Read the complete website
    output = website.read()

    # Return the number of seconds
    # passed since epoch
    close_time = time()

    # Close the website
    website.close()

    if round(close_time-open_time, 2) < 2.5:
        load_status = 'Cepat'
        load_num = 2
    elif round(close_time-open_time, 2) > 2.5 and round(close_time-open_time, 2) < 4:
        load_status = 'Sedang'
        load_num = 1
    elif round(close_time-open_time, 2) > 4:
        load_status = 'Lambat'
        load_num = 0
    # Ganti 'dataset.csv' dengan nama file dataset Anda
    # Returns the Path your .py file is in
    workpath = os.path.dirname(os.path.abspath(__file__))
    seo = open(os.path.join(workpath, 'assets/dataset_v6.csv'), 'rb')
    seo = pd.read_csv(seo)

    # # Pisahkan atribut dan label
    # X = data[['Load Speed', 'Heading Tag Order',
    #           'Meta Title', 'Description', 'OG Image', 'Alt. Text']]
    # y = data['SEO Level']

    # #Mengubah dataFrame ke array Numpy
    # seo = seo.to_numpy()

    # #Membagi Dataset => 80 baris data untuk training dan 20 baris data untuk testing
    # dataTraining = np.concatenate((seo[0:115, :], seo[135:156, :]),
    #                             axis=0)
    # dataTesting = np.concatenate((seo[115:135, :], seo[156:170, :]),
    #                             axis=0)

    # #Memecah Dataset ke Input dan Label
    # inputTraining = dataTraining[:, 0:6]
    # inputTesting = dataTesting[:, 0:6]
    # labelTraining = dataTraining[:,6]
    # labelTesting = dataTesting[:, 6]

    # Fungsi untuk menghitung entropy

    def entropy(data):
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    # # Fungsi untuk menghitung information gain
    def information_gain(data, feature_name, target_name):
        total_entropy = entropy(data[target_name])
        values, counts = np.unique(data[feature_name], return_counts=True)
        weighted_entropy = np.sum((counts / counts.sum()) * [entropy(data.where(
            data[feature_name] == val).dropna()[target_name]) for val in values])
        return total_entropy - weighted_entropy

    # # Fungsi untuk memilih atribut terbaik untuk splitting
    def choose_best_attribute(data, features, target_name):
        information_gains = [information_gain(
            data, feature, target_name) for feature in features]
        return features[np.argmax(information_gains)]

    # # Fungsi untuk membangun pohon keputusan
    def build_decision_tree(data, features, target_name, parent_node_class=None):
        if len(np.unique(data[target_name])) <= 1:
            return np.unique(data[target_name])[0]
        elif len(data) == 0:
            return np.unique(parent_node_class)[0]
        elif len(features) == 0:
            return parent_node_class
        else:
            parent_node_class = np.unique(data[target_name])[np.argmax(
                np.unique(data[target_name], return_counts=True)[1])]
            best_attribute = choose_best_attribute(data, features, target_name)
            tree = {best_attribute: {}}
            features = [i for i in features if i != best_attribute]
            for value in np.unique(data[best_attribute]):
                subset_data = data.where(
                    data[best_attribute] == value).dropna()
                subtree = build_decision_tree(
                    subset_data, features, target_name, parent_node_class)
                tree[best_attribute][value] = subtree
            return tree

    # # Contoh penggunaan algoritma C4.5
    # # Gantilah 'nama_file.csv' dengan nama file dataset Anda dan 'target_column_name' dengan nama kolom target
    # dataset = open(os.path.join(workpath, 'assets/dataset_v6.csv'), 'rb')
    target_column_name = 'Tingkat Optimal'
    features = [col for col in seo.columns if col != target_column_name]

    decision_tree = build_decision_tree(seo, features, target_column_name)

    # # Fungsi untuk membuat prediksi dengan pohon keputusan
    def predict(tree, data):
        for key in tree.keys():
            value = data[key]
            tree = tree[key][value]
            prediction = 0
            if type(tree) is dict:
                prediction = predict(tree, data)
            else:
                prediction = tree
                break
        return prediction

    # # Contoh penggunaan Decision Tree pada data baru
    new_data = {
        'Load Speed': load_num,
        'Heading Tag Order': tag_h1,
        'Meta Title': has_title,
        'Description': status_desc,
        'OG Image': has_og,
        'Alt. Text': has_alt,
    }

    predicted_class = predict(decision_tree, new_data)

    def switch(predicted_class):
        if predicted_class == 4.0:
            return "Optimizing"
        elif predicted_class == 3.0:
            return "Quantitatively"
        elif predicted_class == 2.0:
            return "Managed"
        elif predicted_class == 1.0:
            return "Defined"
        elif predicted_class == 0.0:
            return "Initial"

    # print(f'Predicted Class: {switch(predicted_class)}')

    return render(request, 'diagnose.html', {
        'h1': tag_h1,
        'title': has_title,
        'desc': desc,
        'status_desc': status_desc,
        'image': has_alt,
        'og': has_og,
        'time': round(close_time-open_time, 2),
        'loadstatus': load_status,
        'hasil': switch(predicted_class)})
