# Least Angle Regression (LARS)

The LARS algorithm was introduced in the work of [Efron et al. (2004)](https://arxiv.org/abs/math/0406456). Full version of this paper published in *The Annals of Statistics* can be found [here](paper.pdf).

# Materials

- [Introduction to LARS](https://b-thi.github.io/pdfs/LARS.pdf)
- [Thesis about LARS](https://ir.library.louisville.edu/cgi/viewcontent.cgi?article=3487&context=etd)
- [Presentation about LARS from Stanford](https://hastie.su.domains/TALKS/larstalk.pdf)

# Experiments

1. Synthetic data: iid gaussian predictors, N = 100, p = 10
2. Synthetic data: iid gaussian predictors, N = 100, p = 200
3. Synthetic data: iid gaussian predictors, N = 10_000, p = 100
4. Diabetes dataset.

# To do

- porownac wyniki roznych implementacji lasso z wlasna implementacja lars na kilku zbiorach danych
    - wielkosc wspolczynnikow i liczba niezerowych
    - czas dzialania
    - rozna liczba zmiennych i obserwacji, w szczegolnosci p > n
- sprawdzic sytuacje danych zaleznych
- execution time for each algorithm and each iteration