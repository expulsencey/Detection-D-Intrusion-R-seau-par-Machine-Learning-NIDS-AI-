# Detection-D-Intrusion-R-seau-par-Machine-Learning (NIDS-AI)
Description du projet

Ce projet consiste à concevoir un système de détection d’intrusion réseau (NIDS) basé sur le Machine Learning, capable de distinguer automatiquement un trafic normal d’une activité malveillante.

L’objectif est de remplacer les approches traditionnelles basées uniquement sur des signatures connues par un modèle capable d’apprendre à partir des données et d’identifier des comportements anormaux.

 Le système permet :
	•	d’analyser des connexions réseau
	•	de détecter des attaques (DDoS, scans, comportements suspects)
	•	de produire une prédiction automatique

  Comme décrit dans le rapport, un NIDS surveille en continu le trafic afin d’identifier des activités suspectes ou malveillantes  

Objectifs:
Ce projet poursuit plusieurs objectifs :
	•	Comprendre le fonctionnement des systèmes de détection d’intrusion
	•	Appliquer les techniques de Machine Learning sur des données réelles
	•	Construire un modèle capable de classer le trafic réseau
	•	Déployer une interface interactive pour tester le modèle

 Le projet vise à développer un modèle capable d’identifier efficacement les intrusions réseau tout en renforçant les compétences en Machine Learning

 Dataset:
Le projet utilise un dataset de détection d’intrusion basé sur NSL-KDD.
	•	125 973 lignes
	•	43 caractéristiques réseau
	•	1 variable cible (label)

  
  Méthodologie:
Le projet suit une démarche structurée en plusieurs étapes :

1. Prétraitement des données
	•	Nettoyage des données
	•	Vérification des valeurs manquantes
	•	Encodage des variables catégorielles
	•	Standardisation

Le dataset contient des connexions normales et différentes attaques, permettant d’entraîner le modèle à les distinguer

Cette étape est essentielle car la qualité des données influence directement la performance des modèles 

2. Analyse exploratoire (EDA)
	•	Visualisation des classes (normal vs attaque)
	•	Analyse des distributions
	•	Corrélation entre variables

 Les visualisations montrent clairement que certaines variables permettent de distinguer efficacement les attaques du trafic normal 

 3. Feature Engineering
	•	Sélection des variables les plus pertinentes
	•	Réduction du dataset


4. Entraînement des modèles
Plusieurs algorithmes ont été utilisés :
	•	Régression Logistique
	•	KNN
	•	SVM
	•	Rondom Forest
    •	Naive Bayes
Les modèles ont été évalués à l’aide de métriques comme l’accuracy et la matrice de confusion 


6. Déploiement :
Une application Streamlit a été développée permettant :
	•	la saisie manuelle des données
	•	l’import d’un fichier CSV
	•	la prédiction en temps réel

 L’utilisateur peut tester directement le modèle via une interface interactive

 Fonctionnement du système :
Le pipeline du projet est le suivant :
	1.	Chargement des données
	2.	Prétraitement
	3.	Sélection des features
	4.	Entraînement du modèle
	5.	Prédiction

 Cette architecture suit une progression classique : données → apprentissage → détection

 Remerciements :
Nous tenons à exprimer notre profonde gratitude :
	•	À notre encadrant pour son accompagnement et ses conseils
	•	À l’Université de Djibouti pour les conditions de travail mises à disposition
	•	Aux enseignants pour la qualité de leur formation
	•	Aux membres de l’équipe pour leur collaboration et leur implication

  Bibliographie :
	•	Dataset : Kaggle — Network Intrusion Detection
	•	Scikit-learn Documentation
	•	Python Documentation
	•	Streamlit Documentation
