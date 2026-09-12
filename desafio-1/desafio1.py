"""
desafio1_aluno.py — Desafio 1: Mapa de características para o Perceptron
GBC073 — Inteligência Computacional (FACOM/UFU)

O QUE VOCÊ FAZ: preencher a classe `Submissao` (e só ela).
O QUE O HARNESS FAZ: gera dados, divide 60/40, aplica a sua phi, treina um
Perceptron fixo e mede a acurácia. Rode:  python desafio1_aluno.py

Regras:
  * fit(X) recebe só as entradas de treino, sem rótulos. É opcional.
  * phi(X) transforma (n, d) em (n, d') com d < d' <= 64, de forma determinística.
  * Sem NaN/Inf; e rápido (10 mil pontos em menos de 2 s).
Escore: 0 = igual à identidade (baseline), 100 = igual à referência do professor,
até 125 se superar a referência. Na correção, tarefas OCULTAS da mesma família
substituem estas — não ajuste para um conjunto de dados específico.
"""
import math
import time
import torch
#Douglas Ferreira Martins e Gullit Damião Teixeira de Campos

DIM_MAX = 64
SEMENTES = (0, 1, 2)

# =============================================================================
# >>> SUA SUBMISSÃO — edite apenas esta classe <<<
# =============================================================================
class Submissao:
    DIM_MAX = 64

    # ---- utilitário: blocos ortogonais (Yu et al., 2016 — Orthogonal Random
    # Features). Dentro de cada bloco de tamanho d, as direções são
    # ortogonais entre si; a norma de cada coluna é reamostrada de uma
    # distribuição qui (chi(d)) pra imitar a norma de um vetor gaussiano
    # padrão — isso preserva a mesma distribuição-alvo do RFF clássico,
    # só que com menos redundância entre direções, reduzindo a variância
    # da aproximação do kernel RBF pro mesmo número de features.
    @staticmethod
    def _blocos_ortogonais(gen, d, n, device, dtype):
        n_blocos = math.ceil(n / d)
        cols = []
        for _ in range(n_blocos):
            G = torch.randn(d, d, generator=gen, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(G)
            cols.append(Q)
        U = torch.cat(cols, dim=1)[:, :n]
        normas = torch.sqrt(torch.distributions.Chi2(d).sample((n,)).to(device=device, dtype=dtype))
        return U * normas

    def fit(self, X: torch.Tensor) -> None:
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        d = X.shape[1]
        Xs = (X - self.mu) / self.sd

        rem = self.DIM_MAX - d

        # Reserva mínima pra RFF: a parte "universal" do mapa nunca fica sem
        # espaço, senão o mapa vira só polinomial e perde capacidade de
        # aproximar fronteiras que não sejam quadráticas/produto.
        reserva_rff = max(rem // 4, 4)
        orcamento = max(rem - reserva_rff, 0)
        usado = 0

        # 1) Quadrados individuais (xᵢ²): fronteiras elípticas alinhadas aos
        # eixos. Usa o quanto couber — cai naturalmente com d, sem limiar.
        squares_dim = min(d, orcamento)
        orcamento -= squares_dim; usado += squares_dim
        self.squares_dim = squares_dim

        # 2) Raio² (Σxᵢ²): resolve fronteiras radialmente simétricas em
        # qualquer dimensão. Só entra quando os quadrados individuais NÃO
        # cobrem tudo (squares_dim < d) — quando cobrem, Σxᵢ² é combinação
        # linear do que já existe, e incluí-lo de novo desperdiçaria espaço.
        self.usa_raio = (squares_dim < d) and (orcamento >= 1)
        if self.usa_raio:
            orcamento -= 1; usado += 1

        # 3) Pares cruzados (xᵢxⱼ, i<j): fronteiras tipo produto (ex.: XOR).
        # Crescem em d², então em dimensão alta simplesmente não cabem — é
        # a aritmética do orçamento que os desativa, não um limiar de d.
        idx_i, idx_j = torch.triu_indices(d, d, offset=1)
        n_pares = idx_i.numel()
        self.usa_pares = 0 < n_pares <= orcamento
        if self.usa_pares:
            self.idx_i, self.idx_j = idx_i, idx_j
            orcamento -= n_pares; usado += n_pares

        # 4) RFF ortogonais preenchem o resto do orçamento.
        self.n_rff = max(rem - usado, 0)
        if self.n_rff > 0:
            gen = torch.Generator(device=X.device).manual_seed(0)
            W_dir = self._blocos_ortogonais(gen, d, self.n_rff, X.device, X.dtype)
            self.b = torch.rand(self.n_rff, generator=gen, device=X.device, dtype=X.dtype) * 2 * math.pi

            # Pivô de escala (gamma0) calibrado pela mediana das distâncias
            # entre os próprios dados de treino
            n_sub = min(300, Xs.shape[0])
            sub = Xs[torch.randperm(Xs.shape[0], generator=gen)[:n_sub]]
            d2 = torch.cdist(sub, sub) ** 2
            sigma0 = torch.sqrt(d2[d2 > 0].median() / 2).clamp_min(1e-3)
            gamma0 = 1.0 / sigma0

            # Oitavas geométricas (2⁻² .. 2²) ao redor do pivô: cobre
            # frequências grossas e finas mesmo se o pivô não for exato.
            oitavas = 2.0 ** torch.linspace(-2, 2, 5, device=X.device, dtype=X.dtype)
            rep = oitavas.repeat(math.ceil(self.n_rff / len(oitavas)))[: self.n_rff]

            self.W = W_dir * (gamma0 * rep)
            self._rff_scale = math.sqrt(2.0 / self.n_rff)

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        Xs = ((X - self.mu) / self.sd).clamp(-6.0, 6.0)
        partes = [Xs]

        if self.usa_raio:
            partes.append((Xs ** 2).sum(dim=1, keepdim=True))
        if self.squares_dim > 0:
            partes.append(Xs[:, : self.squares_dim] ** 2)
        if self.usa_pares:
            partes.append(Xs[:, self.idx_i] * Xs[:, self.idx_j])

        if self.n_rff > 0:
            Xs64 = Xs.double()
            proj = Xs64 @ self.W.double() + self.b.double()
            partes.append((torch.cos(proj) * self._rff_scale).to(X.dtype))

        return torch.cat(partes, dim=1)
# =============================================================================
# Harness (não edite daqui para baixo)
# =============================================================================
def _luas(n, g):
    t = torch.rand(n // 2, generator=g) * math.pi
    X = torch.cat([torch.stack([torch.cos(t), torch.sin(t)], 1),
                   torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _circulos(n, g):
    t = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.where(torch.arange(n) < n // 2, 1.0, 0.45)
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return X + 0.08 * torch.randn(n, 2, generator=g), (torch.arange(n) >= n // 2).float()

def _xor(n, g):
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] * X[:, 1] < 0).float()
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _espiral(n, g):
    t = torch.sqrt(torch.rand(n // 2, generator=g)) * 3 * math.pi
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 10
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.cat([a, -a]) + 0.05 * torch.randn(n, 2, generator=g), y

def _esfera(n, g, d=10):
    X = torch.randn(n, d, generator=g)
    r2 = (X ** 2).sum(1)
    return X, (r2 > r2.median()).float()

TAREFAS = {"xor": lambda g: _xor(600, g), "duas_luas": lambda g: _luas(600, g),
           "circulos": lambda g: _circulos(600, g), "espiral": lambda g: _espiral(800, g),
           "esfera_10d": lambda g: _esfera(800, g)}


@torch.no_grad()
def perceptron_pocket(Z, y, epocas=50, eta=1.0, semente=0):
    """Regra de Rosenblatt (w <- w + eta*y*z nos erros) + pocket: guarda o melhor w."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1)], 1)     # viés embutido
    yb = 2 * y - 1
    w = torch.zeros(Zb.shape[1]); melhor_w, melhor_acc = w.clone(), -1.0
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        for i in torch.randperm(len(Zb), generator=g).tolist():
            if yb[i] * (Zb[i] @ w) <= 0:
                w += eta * yb[i] * Zb[i]
        acc = ((Zb @ w) * yb > 0).float().mean().item()
        if acc > melhor_acc:
            melhor_acc, melhor_w = acc, w.clone()
    return melhor_w


def acuracia_balanceada(y, yhat):
    return torch.stack([(yhat[y == c] == c).float().mean() for c in y.unique()]).mean().item()


@torch.no_grad()
def rodar(sub, gerador, semente, checar=True):
    g = torch.Generator().manual_seed(semente)
    X, y = gerador(g)
    idx = torch.randperm(len(X), generator=g); ntr = int(0.6 * len(X))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

    torch.manual_seed(semente)
    sub.fit(Xtr)                                       # nunca recebe ytr
    t0 = time.perf_counter(); Ztr = sub.phi(Xtr); dt = time.perf_counter() - t0
    Zte = sub.phi(Xte)
    if checar:                                         # regras do desafio
        d, dl = Xtr.shape[1], Ztr.shape[1]
        assert Ztr.ndim == 2 and Zte.shape[1] == dl, "phi deve devolver (n, d')"
        assert d < dl <= DIM_MAX, f"exige d < d' <= {DIM_MAX}; recebi d={d}, d'={dl}"
        assert torch.isfinite(Ztr).all() and torch.isfinite(Zte).all(), "NaN/Inf na saída de phi"
        assert torch.allclose(sub.phi(Xtr[:20]), Ztr[:20]), "phi não é determinística"
        assert dt * (10_000 / len(Xtr)) < 2.0, "phi lenta demais (limite: 10^4 pontos em 2 s)"

    w = perceptron_pocket(Ztr, ytr, semente=semente)
    yhat = (torch.cat([Zte, torch.ones(len(Zte), 1)], 1) @ w > 0).float()
    return acuracia_balanceada(yte, yhat)


class _Identidade:                       # baseline (viola d < d', mas é só o ponto zero da escala)
    def fit(self, X): pass
    def phi(self, X): return X

class _RFF:                              # referência: random Fourier features, d' = 64
    def fit(self, X):
        g = torch.Generator().manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
        sigma = math.sqrt(d2[d2 > 0].median().item() / 2)
        self.W = torch.randn(X.shape[1], DIM_MAX, generator=g) / sigma
        self.b = torch.rand(DIM_MAX, generator=g) * 2 * math.pi
    def phi(self, X):
        return math.sqrt(2 / DIM_MAX) * torch.cos(((X - self.mu) / self.sd) @ self.W + self.b)


def _mediana(cls, gerador, checar=True):
    vals = [rodar(cls(), gerador, sem, checar) for sem in SEMENTES]
    return float(torch.tensor(vals).median())


def avaliar():
    print(f"{'tarefa':<12}{'baseline':>10}{'referência':>12}{'você':>8}{'s_t':>7}")
    s = []
    for nome, gen in TAREFAS.items():
        b = _mediana(_Identidade, gen, checar=False)
        r = max(_mediana(_RFF, gen, checar=False), b + 1e-3)
        try:
            m = _mediana(Submissao, gen); erro = ""
        except AssertionError as e:
            m, erro = b, f"   <- {e}"
        st = min(max((m - b) / (r - b), 0.0), 1.25); s.append(st)
        print(f"{nome:<12}{b:>10.3f}{r:>12.3f}{m:>8.3f}{st:>7.2f}{erro}")
    S = 100 * (0.7 * sum(s) / len(s) + 0.3 * min(s))
    print(f"\nESCORE S = {S:.1f}   (0 = baseline, 100 = referência, até 125 com bônus)")
    return S


if __name__ == "__main__":
    avaliar();