"""
exemplo_submissao_d2.py — exemplo de entrega do Desafio 2
Entrega oficial: copie para desafio2/desafio2_nomes.py no seu repositório.
Douglas Ferreira Martins e Gullit Damião Teixeira de Campos

Ideia: em vez de decorar Xavier/He, CALIBRAR o ganho numericamente para a ativação escolhida.
Se z ~ N(0, 1) e W ~ N(0, s^2), a pré-ativação da próxima camada tem variância
    fan_in * s^2 * E[f(z)^2]
Para mantê-la em 1 camada após camada:  s^2 = 1 / (fan_in * E[f(z)^2]).
Troque `ativacao` por outra função e a inicialização se ajusta sozinha.
Rode:  python harness_desafio2.py exemplo_submissao_d2.py --rapido

Referências da API do PyTorch usadas aqui:
  torch.nn.functional.gelu   https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
  torch.Tensor.normal_       https://docs.pytorch.org/docs/stable/generated/torch.Tensor.normal_.html
  torch.Tensor.zero_         https://docs.pytorch.org/docs/stable/generated/torch.Tensor.zero_.html
  torch.no_grad              https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
  torch.Generator            https://docs.pytorch.org/docs/stable/generated/torch.Generator.html
  torch.nn.init (Xavier, He, calculate_gain — para comparar com a conta feita à mão)
                             https://docs.pytorch.org/docs/stable/nn.init.html
  outras ativações: relu, leaky_relu, elu, selu, silu em https://docs.pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
"""
import torch
import torch.nn.init as init
import math

def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

_z = torch.randn(1_000_000, generator=torch.Generator().manual_seed(0))

def calcular_ganho(n_camadas):
    q = q = 0.65 - 0.10 * min(1, 4/n_camadas)
    z = torch.sqrt(torch.tensor(q)) * _z
    ef2 = ativacao(z).pow(2).mean().item()
    return math.sqrt(q / ef2)


@torch.no_grad()
def inicializar(W, b, fan_in, fan_out, camada, n_camadas):
    if camada == 1:
        ganho = 1.0
    else:
        ganho = calcular_ganho(n_camadas)

    init.orthogonal_(W, gain=ganho)

    if camada == n_camadas:
        W.mul_(0.5)

    b.zero_()