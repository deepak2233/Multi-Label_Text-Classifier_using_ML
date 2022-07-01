{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Multi_lablel_classification_Using_ML.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NFvdQOYCGbZu"
      },
      "outputs": [],
      "source": [
        "import numpy as np \n",
        "import pandas as pd \n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IlSppeDpeqTD",
        "outputId": "6d31a809-a923-4e58-c145-36bab2a39186"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "root = '/content/drive/MyDrive/IGP_Assignment/csv/dataset.xlsx'\n",
        "data = pd.read_excel(root)\n",
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "5I1lWoQ2Gglp",
        "outputId": "3f6cc284-c171-4a19-fcca-17349f1dfc29"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                     text  \\\n",
              "0  Customized Romantic Quote Wooden Frame   \n",
              "1    send birthday cake and flower online   \n",
              "2     Someone I Love Birthday Cake (1 kg)   \n",
              "3                                     pig   \n",
              "4                                    bags   \n",
              "\n",
              "                                              target  \n",
              "0   desk accessories,stationery and desk accessories  \n",
              "1  cakes,photo cakes,designer cakes,regular cakes...  \n",
              "2                                  cakes,photo cakes  \n",
              "3                                               none  \n",
              "4  bags and clutches,laptop bags,handbags,sling b...  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f5cf78b7-5c65-4ce9-9ed3-2f45786ff555\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>text</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Customized Romantic Quote Wooden Frame</td>\n",
              "      <td>desk accessories,stationery and desk accessories</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>send birthday cake and flower online</td>\n",
              "      <td>cakes,photo cakes,designer cakes,regular cakes...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Someone I Love Birthday Cake (1 kg)</td>\n",
              "      <td>cakes,photo cakes</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>pig</td>\n",
              "      <td>none</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>bags</td>\n",
              "      <td>bags and clutches,laptop bags,handbags,sling b...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f5cf78b7-5c65-4ce9-9ed3-2f45786ff555')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f5cf78b7-5c65-4ce9-9ed3-2f45786ff555 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f5cf78b7-5c65-4ce9-9ed3-2f45786ff555');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 101
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ORX2x7aGGgnQ",
        "outputId": "8c906de4-62ea-4ea1-b05e-560adad7e38e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(44886, 2)"
            ]
          },
          "metadata": {},
          "execution_count": 102
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def remove_none_row(data):\n",
        "  for idx, ele in enumerate(data['target']):\n",
        "    if ele == 'none':\n",
        "      data = data.drop(idx)\n",
        "  return data\n",
        "\n",
        "def count_none_row(data):\n",
        "  count_none = 0\n",
        "  for i in data['target']:\n",
        "    if i == 'none':\n",
        "      count_none = count_none + 1\n",
        "  return count_none\n",
        "\n",
        "def data_cleaning(PATH):\n",
        "  data = pd.read_excel(PATH)\n",
        "  print(\"data shape\", data.shape)\n",
        "  print(\"Number of NONE value in data\", count_none_row(data))\n",
        "\n",
        "  print(\"----------------After Cleaning data ----------------\")\n",
        "\n",
        "  data = remove_none_row(data)\n",
        "\n",
        "  print(\"data shape\", data.shape)\n",
        "  print(\"Number of NONE value in data\", count_none_row(data))\n",
        "\n",
        "  return data\n"
      ],
      "metadata": {
        "id": "njsit1lJGgqX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = data_cleaning(root)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZmBoK-OOGgsR",
        "outputId": "7ba026b6-1d91-4ca7-d30b-e95033878864"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "data shape (44886, 2)\n",
            "Number of NONE value in data 12923\n",
            "----------------After Cleaning data ----------------\n",
            "data shape (31963, 2)\n",
            "Number of NONE value in data 0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "y8hCm6vhGgxP",
        "outputId": "42d1feb7-2d0c-49da-f82f-5446c8de330e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                     text  \\\n",
              "0  Customized Romantic Quote Wooden Frame   \n",
              "1    send birthday cake and flower online   \n",
              "2     Someone I Love Birthday Cake (1 kg)   \n",
              "4                                    bags   \n",
              "6                         perfume for dad   \n",
              "\n",
              "                                              target  \n",
              "0   desk accessories,stationery and desk accessories  \n",
              "1  cakes,photo cakes,designer cakes,regular cakes...  \n",
              "2                                  cakes,photo cakes  \n",
              "4  bags and clutches,laptop bags,handbags,sling b...  \n",
              "6         beauty and personal care,deos and perfumes  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8b0ed06e-52c9-4300-abbd-2c6bff10d2a5\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>text</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Customized Romantic Quote Wooden Frame</td>\n",
              "      <td>desk accessories,stationery and desk accessories</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>send birthday cake and flower online</td>\n",
              "      <td>cakes,photo cakes,designer cakes,regular cakes...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Someone I Love Birthday Cake (1 kg)</td>\n",
              "      <td>cakes,photo cakes</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>bags</td>\n",
              "      <td>bags and clutches,laptop bags,handbags,sling b...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>perfume for dad</td>\n",
              "      <td>beauty and personal care,deos and perfumes</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8b0ed06e-52c9-4300-abbd-2c6bff10d2a5')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-8b0ed06e-52c9-4300-abbd-2c6bff10d2a5 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-8b0ed06e-52c9-4300-abbd-2c6bff10d2a5');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 105
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def text_cleaning(data):\n",
        "  li = []\n",
        "  for i in data.index:\n",
        "    li.append(data['target'][i].replace(' and ',',')) \n",
        "  data['clean_target'] = li\n",
        "  li1 = []\n",
        "  for i in data.index:\n",
        "    li1.append(data['clean_target'][i].replace(' & ',',')) \n",
        "  data['clean_target'] = li1\n",
        "\n",
        "  return data"
      ],
      "metadata": {
        "id": "7EJ8LF0TSXw8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data = text_cleaning(data)"
      ],
      "metadata": {
        "id": "ulkncpWdTK3n"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "GngNHSeTUFYJ",
        "outputId": "5bbf29d0-faa1-41ec-dc51-ed52d63d070a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                     text  \\\n",
              "0  Customized Romantic Quote Wooden Frame   \n",
              "1    send birthday cake and flower online   \n",
              "2     Someone I Love Birthday Cake (1 kg)   \n",
              "4                                    bags   \n",
              "6                         perfume for dad   \n",
              "\n",
              "                                              target  \\\n",
              "0   desk accessories,stationery and desk accessories   \n",
              "1  cakes,photo cakes,designer cakes,regular cakes...   \n",
              "2                                  cakes,photo cakes   \n",
              "4  bags and clutches,laptop bags,handbags,sling b...   \n",
              "6         beauty and personal care,deos and perfumes   \n",
              "\n",
              "                                        clean_target  \n",
              "0       desk accessories,stationery,desk accessories  \n",
              "1  cakes,photo cakes,designer cakes,regular cakes...  \n",
              "2                                  cakes,photo cakes  \n",
              "4  bags,clutches,laptop bags,handbags,sling bags,...  \n",
              "6                 beauty,personal care,deos,perfumes  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-c4cacad5-d6e2-4679-a03f-7e9bb1ab76ea\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>text</th>\n",
              "      <th>target</th>\n",
              "      <th>clean_target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Customized Romantic Quote Wooden Frame</td>\n",
              "      <td>desk accessories,stationery and desk accessories</td>\n",
              "      <td>desk accessories,stationery,desk accessories</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>send birthday cake and flower online</td>\n",
              "      <td>cakes,photo cakes,designer cakes,regular cakes...</td>\n",
              "      <td>cakes,photo cakes,designer cakes,regular cakes...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Someone I Love Birthday Cake (1 kg)</td>\n",
              "      <td>cakes,photo cakes</td>\n",
              "      <td>cakes,photo cakes</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>bags</td>\n",
              "      <td>bags and clutches,laptop bags,handbags,sling b...</td>\n",
              "      <td>bags,clutches,laptop bags,handbags,sling bags,...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>perfume for dad</td>\n",
              "      <td>beauty and personal care,deos and perfumes</td>\n",
              "      <td>beauty,personal care,deos,perfumes</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-c4cacad5-d6e2-4679-a03f-7e9bb1ab76ea')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-c4cacad5-d6e2-4679-a03f-7e9bb1ab76ea button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-c4cacad5-d6e2-4679-a03f-7e9bb1ab76ea');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 108
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import re\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.preprocessing import MultiLabelBinarizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.model_selection import learning_curve\n",
        "from sklearn.model_selection import ShuffleSplit\n",
        "from sklearn.dummy import DummyClassifier\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.linear_model import SGDClassifier\n",
        "from sklearn.svm import LinearSVC\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.multiclass import OneVsRestClassifier\n",
        "from sklearn import model_selection\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "from sklearn.metrics import make_scorer\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.metrics import hamming_loss\n",
        "from sklearn import metrics\n",
        "from sklearn.metrics import f1_score, precision_score, recall_score"
      ],
      "metadata": {
        "id": "DK8Z-X33WFoC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def clean_text(text):\n",
        "    text = text.lower()\n",
        "    text = text.strip(' ')\n",
        "    return text"
      ],
      "metadata": {
        "id": "vmzhgSslWFrW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['text'] = data['text'].apply(lambda x: clean_text(x))"
      ],
      "metadata": {
        "id": "BxAUUjiFXjmU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data['text']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c6NFvLyAh9LK",
        "outputId": "4a1b6fd1-b8e3-40b6-f386-7f6d70ad37aa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0                   customized romantic quote wooden frame\n",
              "1                     send birthday cake and flower online\n",
              "2                      someone i love birthday cake (1 kg)\n",
              "4                                                     bags\n",
              "6                                          perfume for dad\n",
              "                               ...                        \n",
              "44880                   dream big personalized photo frame\n",
              "44881                           personalised led key chain\n",
              "44882                    online flower delivery in gurgaon\n",
              "44883    celestial vanilla almond cake (eggless) (half kg)\n",
              "44885                    set of 3 personalized photo frame\n",
              "Name: text, Length: 31963, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 112
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import ast\n",
        "data['clean_target'] = data['clean_target'].apply(lambda x : (x.split(\",\")))"
      ],
      "metadata": {
        "id": "4ZjUNDBTUFK4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "AMstqfjibiKL",
        "outputId": "646436cd-c29e-435c-a341-e1bfc1e831a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                     text  \\\n",
              "0  customized romantic quote wooden frame   \n",
              "1    send birthday cake and flower online   \n",
              "2     someone i love birthday cake (1 kg)   \n",
              "4                                    bags   \n",
              "6                         perfume for dad   \n",
              "\n",
              "                                              target  \\\n",
              "0   desk accessories,stationery and desk accessories   \n",
              "1  cakes,photo cakes,designer cakes,regular cakes...   \n",
              "2                                  cakes,photo cakes   \n",
              "4  bags and clutches,laptop bags,handbags,sling b...   \n",
              "6         beauty and personal care,deos and perfumes   \n",
              "\n",
              "                                        clean_target  \n",
              "0   [desk accessories, stationery, desk accessories]  \n",
              "1  [cakes, photo cakes, designer cakes, regular c...  \n",
              "2                               [cakes, photo cakes]  \n",
              "4  [bags, clutches, laptop bags, handbags, sling ...  \n",
              "6            [beauty, personal care, deos, perfumes]  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-6fdaa919-61f7-436d-9147-eff421dcf72b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>text</th>\n",
              "      <th>target</th>\n",
              "      <th>clean_target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>customized romantic quote wooden frame</td>\n",
              "      <td>desk accessories,stationery and desk accessories</td>\n",
              "      <td>[desk accessories, stationery, desk accessories]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>send birthday cake and flower online</td>\n",
              "      <td>cakes,photo cakes,designer cakes,regular cakes...</td>\n",
              "      <td>[cakes, photo cakes, designer cakes, regular c...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>someone i love birthday cake (1 kg)</td>\n",
              "      <td>cakes,photo cakes</td>\n",
              "      <td>[cakes, photo cakes]</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>bags</td>\n",
              "      <td>bags and clutches,laptop bags,handbags,sling b...</td>\n",
              "      <td>[bags, clutches, laptop bags, handbags, sling ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>perfume for dad</td>\n",
              "      <td>beauty and personal care,deos and perfumes</td>\n",
              "      <td>[beauty, personal care, deos, perfumes]</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-6fdaa919-61f7-436d-9147-eff421dcf72b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-6fdaa919-61f7-436d-9147-eff421dcf72b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-6fdaa919-61f7-436d-9147-eff421dcf72b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 114
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = data['text']\n",
        "y = data['clean_target']"
      ],
      "metadata": {
        "id": "TcFV9LtUj_tC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import warnings\n",
        "import logging\n",
        "from scipy.sparse import hstack\n",
        "warnings.filterwarnings(\"ignore\")\n"
      ],
      "metadata": {
        "id": "7MBR-r7YXtbx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "multilabel_binarizer = MultiLabelBinarizer()\n",
        "y_bin = multilabel_binarizer.fit_transform(y)"
      ],
      "metadata": {
        "id": "QwCtQKDNXv8X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_bin"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DcBsOFYubCQk",
        "outputId": "9fc5256a-443a-4c35-d4bb-ce8c6d5cffa2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       ...,\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0],\n",
              "       [0, 0, 0, ..., 0, 0, 0]])"
            ]
          },
          "metadata": {},
          "execution_count": 118
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "multilabel_binarizer.classes_"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5iDuVMoMbtIp",
        "outputId": "1f522845-d0c2-4f5e-c729-03260f88ce99"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['accessories', 'action figures', 'animals', 'apparel',\n",
              "       'appliances', 'artificial plants', 'assembly toys', 'baby',\n",
              "       'baby care', 'backpacks', 'bags', 'bamboo plants', 'bangles',\n",
              "       'bar accessories', 'barware', 'basket arrangements', 'bath items',\n",
              "       'beauty', 'bed covers', 'bedsheets', 'belts', 'beverages',\n",
              "       'birds paintings', 'biscuits', 'blooming plants', 'board games',\n",
              "       'bonsai', 'bouquets', 'bouquets bunches', 'bows', 'bracelets',\n",
              "       'branded chocolates', 'bunches', 'cakes', 'calendars', 'candies',\n",
              "       'candles', 'carriers', 'cheese cakes', 'chess', 'chocolates',\n",
              "       'clocks', 'clubs', 'clutches', 'cookies', 'covers', 'covers ',\n",
              "       'creative play', 'cufflinks', 'cupcakes', 'cups', 'cushions',\n",
              "       'decor', 'deos', 'designer cakes', 'desk accessories', 'diaries',\n",
              "       'dining', 'dolls', 'dress materials', 'dresses', 'dry cakes',\n",
              "       'dry fruits', 'dryfruits', 'earrings', 'educational games',\n",
              "       'electronic toys', 'electronics', 'ethnic paintings',\n",
              "       'ethnic skirts', 'ethnic wear', 'ethnic wears', 'exercise',\n",
              "       'face mask', 'fashion accessories', 'feeding', 'fitness',\n",
              "       'flowers', 'fragrances', 'furnishings', 'furniture',\n",
              "       'furniture decor', 'games', 'general shopping', 'gift cards',\n",
              "       'gifts cards', 'glasses', 'gourmet baskets', 'green plants',\n",
              "       'hair accessories', 'hair care', 'handbags', 'handbags clutches',\n",
              "       'handmade chocolates', 'health', 'health hamper', 'health hampers',\n",
              "       'holders', 'holidays', 'home', 'home decor', 'home décor',\n",
              "       'indoor games', 'indoor plants', 'infants', 'interactive toys',\n",
              "       'jewellery', 'jewellery boxes', 'jhumkas', 'jugs', 'key chains',\n",
              "       'kids', 'kids clothing', 'kids fashion accessories',\n",
              "       'kids furnishings', 'kids school supplies', 'kids special',\n",
              "       'kids watches', 'kitchen', 'kitchen accessories', 'kits', 'kurtis',\n",
              "       'lamps', 'lanterns', 'laptop bags', 'learning', 'love soft toys',\n",
              "       'makeup items', 'men western wear', 'mirrors',\n",
              "       'mobile accessories', 'modern paintings', 'mousse cakes', 'mugs',\n",
              "       'musical toys', 'namkeens', 'nautical pieces', 'necklaces',\n",
              "       'nursing', 'outdoor games', 'paintings', 'pen sets', 'pendants',\n",
              "       'perfumes', 'personal care', 'personalized plants', 'photo cakes',\n",
              "       'photo frames', 'planners', 'platters', 'playsets',\n",
              "       'plush pillows', 'posters', 'pots', 'prints', 'puaj accessories',\n",
              "       'puja accessories', 'pull back toys', 'puzzles',\n",
              "       'puzzles assembly toys', 'quilts', 'rattles', 'regular cakes',\n",
              "       'religious', 'restaurants', 'rings', 'role play sets',\n",
              "       'salwar suits', 'sarees', 'sets', 'shaving items', 'shawls',\n",
              "       'shirts', 'showpieces', 'skin care', 'sling bags', 'snacks',\n",
              "       'soft toys', 'spiritual idols', 'spiritual paintings', 'sports',\n",
              "       'sports accessories', 'stationery', 'stoles', 'strollers',\n",
              "       'stuffed animals', 'stuffed characters', 'succulents',\n",
              "       'sugarfree sweets', 'suits', 'sweaters', 'sweets', 'table decor',\n",
              "       'table games', 'tableware', 'teddy bear', 'teddy bears', 'tees',\n",
              "       'teethers', 'ties', 'toddlers', 'toffies', 'tools', 'tops',\n",
              "       'toy vehicles', 'travel', 'trays', 'tshirts', 'vases',\n",
              "       'vehicle toys', 'wall decor', 'wallets', 'watches',\n",
              "       'wedding chudas', 'women western wear', 'wrist bands'],\n",
              "      dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 119
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pd.DataFrame(y_bin, columns=multilabel_binarizer.classes_) "
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 522
        },
        "id": "Cpnzu5Jab04O",
        "outputId": "068b8f7f-532a-4016-89ba-898c3b536de3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       accessories  action figures  animals  apparel  appliances  \\\n",
              "0                0               0        0        0           0   \n",
              "1                0               0        0        0           0   \n",
              "2                0               0        0        0           0   \n",
              "3                0               0        0        0           0   \n",
              "4                0               0        0        0           0   \n",
              "...            ...             ...      ...      ...         ...   \n",
              "31958            0               0        0        0           0   \n",
              "31959            0               0        0        0           0   \n",
              "31960            0               0        0        0           0   \n",
              "31961            0               0        0        0           0   \n",
              "31962            0               0        0        0           0   \n",
              "\n",
              "       artificial plants  assembly toys  baby  baby care  backpacks  ...  \\\n",
              "0                      0              0     0          0          0  ...   \n",
              "1                      0              0     0          0          0  ...   \n",
              "2                      0              0     0          0          0  ...   \n",
              "3                      0              0     0          0          1  ...   \n",
              "4                      0              0     0          0          0  ...   \n",
              "...                  ...            ...   ...        ...        ...  ...   \n",
              "31958                  0              0     0          0          0  ...   \n",
              "31959                  0              0     0          0          0  ...   \n",
              "31960                  0              0     0          0          0  ...   \n",
              "31961                  0              0     0          0          0  ...   \n",
              "31962                  0              0     0          0          0  ...   \n",
              "\n",
              "       trays  tshirts  vases  vehicle toys  wall decor  wallets  watches  \\\n",
              "0          0        0      0             0           0        0        0   \n",
              "1          0        0      0             0           0        0        0   \n",
              "2          0        0      0             0           0        0        0   \n",
              "3          0        0      0             0           0        0        0   \n",
              "4          0        0      0             0           0        0        0   \n",
              "...      ...      ...    ...           ...         ...      ...      ...   \n",
              "31958      0        0      0             0           0        0        0   \n",
              "31959      0        0      0             0           0        0        0   \n",
              "31960      0        0      0             0           0        0        0   \n",
              "31961      0        0      0             0           0        0        0   \n",
              "31962      0        0      0             0           0        0        0   \n",
              "\n",
              "       wedding chudas  women western wear  wrist bands  \n",
              "0                   0                   0            0  \n",
              "1                   0                   0            0  \n",
              "2                   0                   0            0  \n",
              "3                   0                   0            0  \n",
              "4                   0                   0            0  \n",
              "...               ...                 ...          ...  \n",
              "31958               0                   0            0  \n",
              "31959               0                   0            0  \n",
              "31960               0                   0            0  \n",
              "31961               0                   0            0  \n",
              "31962               0                   0            0  \n",
              "\n",
              "[31963 rows x 216 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b337dc84-222f-4958-9c2f-ece957c40902\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>accessories</th>\n",
              "      <th>action figures</th>\n",
              "      <th>animals</th>\n",
              "      <th>apparel</th>\n",
              "      <th>appliances</th>\n",
              "      <th>artificial plants</th>\n",
              "      <th>assembly toys</th>\n",
              "      <th>baby</th>\n",
              "      <th>baby care</th>\n",
              "      <th>backpacks</th>\n",
              "      <th>...</th>\n",
              "      <th>trays</th>\n",
              "      <th>tshirts</th>\n",
              "      <th>vases</th>\n",
              "      <th>vehicle toys</th>\n",
              "      <th>wall decor</th>\n",
              "      <th>wallets</th>\n",
              "      <th>watches</th>\n",
              "      <th>wedding chudas</th>\n",
              "      <th>women western wear</th>\n",
              "      <th>wrist bands</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>31958</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>31959</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>31960</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>31961</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>31962</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>31963 rows × 216 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b337dc84-222f-4958-9c2f-ece957c40902')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-b337dc84-222f-4958-9c2f-ece957c40902 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-b337dc84-222f-4958-9c2f-ece957c40902');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 120
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X,y_bin, test_size = 0.2, random_state = 0)"
      ],
      "metadata": {
        "id": "d7IRXhBAkQYf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vectorizer_char_tfidf = TfidfVectorizer(analyzer = 'char_wb',\n",
        "                                       stop_words='english',\n",
        "                                       ngram_range=(1,3),\n",
        "                                       max_features=5000)"
      ],
      "metadata": {
        "id": "GY8boEqAb25c"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_train_char = vectorizer_char_tfidf.fit(X_train)\n",
        "X_tfidf_train_char = vectorizer_char_tfidf.transform(X_train)\n",
        "X_tfidf_test_char = vectorizer_char_tfidf.transform(X_test)\n"
      ],
      "metadata": {
        "id": "WwmOcotpnI5K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_test_char.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cCCkrQ90qnOQ",
        "outputId": "6070991d-2db0-4a81-ca17-2866960afba6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(6393, 5000)"
            ]
          },
          "metadata": {},
          "execution_count": 152
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_train_char.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gU40J6ZpqkfS",
        "outputId": "b41df3ff-5282-41a3-a551-2cbbd5706a08"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(25570, 5000)"
            ]
          },
          "metadata": {},
          "execution_count": 151
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "vectorizer_word_tfidf = TfidfVectorizer(analyzer = 'word',\n",
        "                                       stop_words='english',\n",
        "                                       ngram_range=(1,3),\n",
        "                                       max_features=5000)"
      ],
      "metadata": {
        "id": "0NFnsbuzksbA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_train_word = vectorizer_word_tfidf.fit(X_train)\n",
        "X_tfidf_train_word = vectorizer_word_tfidf.transform(X_train)\n",
        "X_tfidf_test_word = vectorizer_word_tfidf.transform(X_test)\n"
      ],
      "metadata": {
        "id": "4Tn2QRaisc64"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_train_word.shape, X_tfidf_test_word.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CRSwtiJvgEvM",
        "outputId": "f98675fb-b170-4ba7-e947-d05ae3bdceab"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((25570, 5000), (6393, 5000))"
            ]
          },
          "metadata": {},
          "execution_count": 159
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "vectorizer_ch_tfidf = TfidfVectorizer(analyzer = 'char',\n",
        "                                       stop_words='english',\n",
        "                                       ngram_range=(1,3),\n",
        "                                       max_features=5000)"
      ],
      "metadata": {
        "id": "eANPkTwsvrsj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_train_ch = vectorizer_ch_tfidf.fit(X_train)\n",
        "X_tfidf_train_ch = vectorizer_ch_tfidf.transform(X_train)\n",
        "X_tfidf_test_ch = vectorizer_ch_tfidf.transform(X_test)\n"
      ],
      "metadata": {
        "id": "9ULgaVpMvzqj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X_tfidf_train_ch.shape, X_tfidf_test_ch.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "U7li0-u0vzzM",
        "outputId": "9fe9f6e5-2458-4fd8-8a4d-568b063276b0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((25570, 5000), (6393, 5000))"
            ]
          },
          "metadata": {},
          "execution_count": 163
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def avg_jacard(y_true,y_pred):\n",
        "    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)    \n",
        "    return jacard.mean()*100\n",
        "    \n",
        "def print_score(y_pred, clf):\n",
        "    print(\"Clf: \", clf.__class__.__name__)\n",
        "    print(\"Jacard score: {}\".format(avg_jacard(y_test, y_pred)))\n",
        "    print(\"Hamming loss: {}\".format(hamming_loss(y_pred, y_test)*100))\n",
        "    print(\"---\")"
      ],
      "metadata": {
        "id": "lLb2DdSJgE0a"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lr = LogisticRegression(solver = 'lbfgs')\n",
        "svc = LinearSVC()\n",
        "sgd = SGDClassifier()\n",
        "rfc = RandomForestClassifier()"
      ],
      "metadata": {
        "id": "T7NpMeYLkAXh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Analyzer -> Char_wb with 5000 max_featur"
      ],
      "metadata": {
        "id": "kvcxN1PIwEj4"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for classfier in [lr, svc,  sgd, rfc]:\n",
        "  clf = OneVsRestClassifier(classfier)\n",
        "  clf.fit(X_tfidf_train_char, y_train)\n",
        "  y_pred = clf.predict(X_tfidf_test_char)\n",
        "  print_score(y_pred, classfier)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h4yfUdANdfuk",
        "outputId": "f2d70fb7-e6e9-438a-b728-ce9f3888444b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Clf:  LogisticRegression\n",
            "Jacard score: 75.06695070749505\n",
            "Hamming loss: 0.3637514411016679\n",
            "---\n",
            "Clf:  LinearSVC\n",
            "Jacard score: 87.14136346820534\n",
            "Hamming loss: 0.2096477049550724\n",
            "---\n",
            "Clf:  SGDClassifier\n",
            "Jacard score: 79.38823160432733\n",
            "Hamming loss: 0.308207472293191\n",
            "---\n",
            "Clf:  RandomForestClassifier\n",
            "Jacard score: 80.0035330173763\n",
            "Hamming loss: 0.289596259798043\n",
            "---\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Analyzer -> word with 5000 max_featur"
      ],
      "metadata": {
        "id": "Ru36PTBLwQ1-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for classfier in [lr, svc,  sgd, rfc]:\n",
        "  clf = OneVsRestClassifier(classfier)\n",
        "  clf.fit(X_tfidf_train_word, y_train)\n",
        "  y_pred = clf.predict(X_tfidf_test_word)\n",
        "  print_score(y_pred, classfier)"
      ],
      "metadata": {
        "id": "Don8Ik-5x7LF",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ba8393ac-d3fa-4ac1-88b2-cbc5c41d367f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Clf:  LogisticRegression\n",
            "Jacard score: 75.54225468467608\n",
            "Hamming loss: 0.34738516085301635\n",
            "---\n",
            "Clf:  LinearSVC\n",
            "Jacard score: 85.71669128098034\n",
            "Hamming loss: 0.22434839031116208\n",
            "---\n",
            "Clf:  SGDClassifier\n",
            "Jacard score: 82.04678457611352\n",
            "Hamming loss: 0.2657710111174838\n",
            "---\n",
            "Clf:  RandomForestClassifier\n",
            "Jacard score: 83.64599154603847\n",
            "Hamming loss: 0.25280833782319784\n",
            "---\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Analyzer -> Char with 5000 max_featur"
      ],
      "metadata": {
        "id": "GnrrUNQWwWJw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for classfier in [lr, svc,  sgd, rfc]:\n",
        "  clf = OneVsRestClassifier(classfier)\n",
        "  clf.fit(X_tfidf_train_ch, y_train)\n",
        "  y_pred = clf.predict(X_tfidf_test_ch)\n",
        "  print_score(y_pred, classfier)"
      ],
      "metadata": {
        "id": "5aCHWtKwx7OI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d07e77d1-9205-4bd5-8d32-2c3445d4bf65"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Clf:  LogisticRegression\n",
            "Jacard score: 74.50785705595652\n",
            "Hamming loss: 0.36939998030252996\n",
            "---\n",
            "Clf:  LinearSVC\n",
            "Jacard score: 87.08210346408377\n",
            "Hamming loss: 0.20928561910886329\n",
            "---\n",
            "Clf:  SGDClassifier\n",
            "Jacard score: 79.50345841828714\n",
            "Hamming loss: 0.30407969364640725\n",
            "---\n",
            "Clf:  RandomForestClassifier\n",
            "Jacard score: 79.04023475351491\n",
            "Hamming loss: 0.2997346634918979\n",
            "---\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "I've compared with three analyzer in order to increase the accuracy. Hence, The best accuracy has found 87.1 with \"char_wb\""
      ],
      "metadata": {
        "id": "zSFPWQjVy104"
      }
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "AszYQH9bzWqK"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
