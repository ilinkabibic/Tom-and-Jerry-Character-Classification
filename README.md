# Tom and Jerry Character Classification

Projekat iz predmeta **Mašinsko učenje**.

## Članovi tima

- **Ilinka Bibić (Ika)**
- **Blagoje Rozgić**

---

## Opis projekta

Cilj projekta je klasifikacija likova iz crtanog filma *Tom i Jerry* u 4 klase: **tom**, **jerry**, **both** (oba lika na slici) i **neither** (nijedan).

Implementirana su i upoređena četiri pristupa:

1. **SVM** (RBF kernel) nad ručno dizajniranim osobinama (HOG + histogram boja)
2. **Random Forest** nad istim osobinama
3. **CNN** treniran od nule
4. **Transfer learning** sa ResNet18 (pretreniran na ImageNet skupu)

Dodatno je analizirana interpretabilnost modela pomoću **Grad-CAM** tehnike, i ponašanje modela na skupu posebno teških/graničnih primera (`challenges.csv`), koji je namerno isključen iz treninga i korišćen kao pravi held-out skup.

---

## Skup podataka

Korišćen je **Tom and Jerry Image Classification** dataset sa Kaggle-a:
https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification

Dataset sadrži 5478 slika, podeljenih u 4 klase (tom, jerry, both, neither), i poseban fajl `challenges.csv` sa 32 slike koje je autor dataset-a označio kao vizuelno teške/granične primere.

Podela na train/val/test je **blok-bazirana i stratifikovana** — susedni frejmovi iz istog video isečka grupisani su u blokove od po 50, tako da ceo blok ide ili u train, ili u val, ili u test (nikad razdvojen preko granice), koristeći `random_state=42`.

**Svih 32 slike iz `challenges.csv` je namerno isključeno** iz train/val/test podele (`split == 'excluded'`), kako bi predstavljale pravi held-out skup za analizu u notebook-u 07 — nijedan model ih nije video tokom treninga.

| Skup | Broj slika |
|---|---:|
| Train | 3755 |
| Validation | 847 |
| Test | 844 |
| Excluded (challenges.csv) | 32 |
| **Ukupno** | **5478** |

Slike nisu uključene direktno u repozitorijum zbog veličine — preuzimaju se sa Kaggle-a u okviru svakog notebook-a.

---

## Struktura repozitorijuma

```text
Tom-and-Jerry-Character-Classification/
│
├── notebooks/
│   ├── 01_eda_split.ipynb
│   ├── 02_klasican_ml_ika.ipynb
│   ├── 03_klasican_ml_blagoje.ipynb
│   ├── 04_transfer_learning_ika.ipynb
│   ├── 05_cnn_blagoje.ipynb
│   ├── 06_poredjenje_modela_blagoje.ipynb
│   ├── 07_challenges.ipynb
│   └── 08_demo.ipynb
│
├── models/
│   ├── svm_final.joblib
│   ├── svm_scaler.joblib
│   ├── rf_final.joblib
│   ├── resnet18_transfer.pt
│   └── cnn_scratch.pt
│
├── demo/
│   └── frame1794.jpg
│
├── data_split.csv
├── features_hog_color.npy
└── README.md
```

---

## Notebook-i

### 01 — EDA i podela podataka
Učitavanje dataset-a, pregled klasa, formiranje blok-bazirane stratifikovane train/val/test podele, i isključivanje 32 slike iz `challenges.csv` iz te podele (kao pravi held-out skup). Rezultat se čuva u `data_split.csv`.

### 02 — Klasičan ML: SVM (Ika)
Ekstrakcija osobina (HOG + HSV histogram boja, 1860 osobina po slici, čuvaju se u `features_hog_color.npy`), standardizacija (StandardScaler), SVM sa RBF kernelom, podešavanje hiperparametara (GridSearchCV).

### 03 — Klasičan ML: Random Forest (Blagoje)
Random Forest nad istim osobinama (HOG + boja), ručna pretraga hiperparametara na validacionom skupu.

### 04 — Transfer learning: ResNet18 (Ika)
ResNet18 pretreniran na ImageNet-u, zamenjen poslednji (fully-connected) sloj za 4 klase, fine-tuning na našem dataset-u.

### 05 — CNN od nule (Blagoje)
Konvoluciona mreža trenirana isključivo na našim podacima, bez pretreniranih težina.

### 06 — Poređenje modela (Blagoje)
Učitavanje sva 4 sačuvana modela i uporedna evaluacija na istom test skupu (844 slike), bez ponovnog treniranja.

### 07 — Analiza na `challenges.csv`
Evaluacija sva 4 sačuvana modela na 32 slike iz `challenges.csv` (pravi held-out skup). Uključuje poređenje po accuracy/macro-F1, pregled po pojedinačnim slikama (koliko modela je pogodilo svaku), i vizuelni prikaz najtežih i najlakših primera.

### 08 — Demo (za odbranu)
Live klasifikacija jedne slike (`demo/frame1794.jpg`) koristeći sva 4 modela, plus Grad-CAM vizuelizacija (gde model "gleda") za CNN i ResNet18.

---

## Rezultati (test skup, 844 slike)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| SVM (RBF) | 46.68% | 0.4458 |
| Random Forest | 50.47% | 0.4223 |
| CNN (od nule) | 59.00% | 0.5537 |
| **ResNet18 (transfer learning)** | **83.41%** | **0.7940** |

ResNet18 transfer learning je ubedljivo najbolji model, zahvaljujući težinama pretreniranim na ImageNet-u — sa relativno malo naših podataka postiže znatno bolju generalizaciju od modela treniranih od nule.

---

## Sačuvani modeli

Svi finalni modeli su sačuvani u `models/` folderu i mogu se direktno učitati (bez ponovnog treniranja):

- `svm_final.joblib` + `svm_scaler.joblib`
- `rf_final.joblib`
- `resnet18_transfer.pt`
- `cnn_scratch.pt`

---

## Kako pokrenuti

Svi notebook-i su pravljeni za **Google Colab**. Preporučeni redosled: `01` → `02`/`03`/`04`/`05` (nezavisni jedni od drugih) → `06` → `07` → `08`.

Potrebni Colab Secrets (`userdata`):
- `KAGGLE_API_TOKEN` — za preuzimanje dataset-a sa Kaggle-a
- `GITHUB_TOKEN` — samo za notebook-e koji nešto čuvaju/pushuju (01–05)

---

## Korišćene tehnologije

Python, scikit-learn, PyTorch, torchvision, OpenCV, scikit-image, pandas, matplotlib, seaborn.

---

## Podela rada

### Ilinka Bibić (Ika)
- EDA i formiranje train/val/test podele (01), uključujući ispravku podele da isključi `challenges.csv` slike iz treninga
- implementacija klasičnog ML pristupa — SVM (02): ekstrakcija osobina, standardizacija, GridSearchCV
- implementacija transfer learning pristupa — ResNet18 (04)
- čuvanje i verifikacija svih sačuvanih modela
- analiza na `challenges.csv` (07): evaluacija sva 4 modela na held-out skupu, pregled po slikama
- demo notebook za odbranu (08), uključujući Grad-CAM vizuelizaciju

### Blagoje Rozgić
- implementacija klasičnog ML pristupa — Random Forest (03): treniranje nad zajedničkim
  osobinama (HOG + histogram boja), ručno podešavanje hiperparametara na validacionom skupu
  (broj stabala, `min_samples_leaf`, `class_weight='balanced'`)
- implementacija dubokog modela — CNN od nule (05): arhitektura konvolucione mreže
  (3 konvoluciona bloka + potpuno povezani slojevi, dropout), augmentacija podataka,
  class weights zbog neizbalansiranih klasa, čuvanje najboljeg modela po val macro-F1
- finalna TEST evaluacija svojih modela (RF i CNN) istom deljenom funkcijom `evaluate_on_test`
- čuvanje finalnih modela `rf_final.joblib` i `cnn_scratch.pt` u `models/`
- poređenje modela (06): učitavanje sva 4 sačuvana modela i uporedna evaluacija na istom
  test skupu (844 slike), uporedna tabela i grafik (bez ponovnog treniranja)
- doprinos zajedničkim delovima: korišćenje zajedničke podele (`data_split.csv`) i osobina
  (`features_hog_color.npy`), i dosledna evaluacija (macro-F1) radi poštenog poređenja

README, završna analiza projekta i priprema za odbranu predstavljaju zajednički deo rada.

---

## Zaključak

## Zaključak

Zanimljivo, poredak modela nije isti na svakom podskupu podataka: dok je na punom test skupu ResNet18 ubedljivo ispred CNN-a (macro-F1 0.79 naspram 0.55),
na skupu iz `challenges.csv` (07) CNN ga neznatno nadmašuje (macro-F1 0.87 naspram 0.85, računato za klase koje se stvarno pojavljuju u tom skupu).

Grad-CAM analiza (08) delimično objašnjava zašto: ResNet18 se pri odluci jasno fokusira na sam lik, dok se CNN treniran od nule delom oslanja i na pozadinu/ivice slike.
