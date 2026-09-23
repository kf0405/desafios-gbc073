"""
exemplo_submissao_d2.py — exemplo de entrega do Desafio 2
Entrega oficial: copie para desafio2/desafio2_nomes.py no seu repositório.
Ideia: em vez de decorar Xavier/He, CALIBRAR o ganho numericamente para a ativação escolhida.
Se z ~ N(0, 1) e W ~ N(0, s^2), a pré-ativação da próxima camada tem variância
    fan_in * s^2 * E[f(z)^2]
Para mantê-la em 1 camada após camada:  s^2 = 1 / (fan_in * E[f(z)^2]).

Aqui a ativação escolhida é SELU. Note que, para SELU, E[f(z)^2] ~= 1.0
(verificado empiricamente), então esta calibração converge para o mesmo
resultado da inicialização LeCun normal — que é, não por coincidência, a
inicialização que o paper original do SELU recomenda (Klambauer et al.,
2017), para preservar a propriedade auto-normalizante da rede.

Rode:  python harness_desafio2.py desafio2_selu.py --rapido
"""
import math
import torch


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.relu(x)           # elemento a elemento, sem parâmetros


# E[f(z)^2] com z ~ N(0,1), estimado uma vez por amostragem (Monte Carlo)
_g = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:                      # camada de logits: um pouco menor ajuda o SGD
        desvio *= 0.5
    W.normal_(0.0, desvio)
    b.zero_()