# campylaspis

Estrutura inicial para um projeto Python de pesquisa.

Estrutura principal:
- `src/campylaspis/` — código fonte do pacote
- `configs/` — exemplos de configuração em YAML
- `scripts/` — scripts utilitários (diretório mantido vazio com .gitkeep)

Instalação de desenvolvimento:
```bash
pip install -e .
```

Exemplo mínimo de uso:
```python
from campylaspis import info
print(info())
```

## Geração de Dados Sintéticos

O projeto inclui um pipeline completo para geração de imagens sintéticas de espécimes científicos usando Stable Diffusion com ControlNet para garantir fidelidade estrutural.

### Modos de Geração

- **text_only**: Geração baseada apenas em prompts de texto (modo padrão)
- **structure_guided**: Geração guiada por estrutura usando ControlNet com detecção de bordas de ilustrações científicas
- **body_part**: Geração especializada para partes específicas do corpo, detectadas automaticamente do nome do arquivo

### Modo Body Part

O modo `body_part` detecta automaticamente a parte do corpo a partir do nome do arquivo e gera prompts especializados:

- `species_p1.png` → `pereopod_1` (primeiro pereópodo)
- `species_carapace.png` → `carapace` (carapaça)
- `species_uropod.png` → `uropod` (urópodo)
- `species_lateral_body.png` → `lateral_body` (vista lateral do corpo)

**Exemplo de uso:**
```python
metadata = generate_dataset_from_taxonomy(
    illustration_paths=["data/illustrations/species_p1.png"],
    description_texts=["First pereopod with specialized setae"],
    output_dir="generated_body_parts",
    generation_mode="body_part",
    structure_strength=0.8
)
```

**Prompt gerado automaticamente:**
```
"high-resolution macro photograph of crustacean first pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure, professional scientific photography, high detail, sharp focus, uniform lighting, taxonomic reference quality"
```
### Appearance Prior Dataset

O sistema inclui suporte para um dataset de aparência prévia (`AppearancePrior`) que guia coloração e textura realistas durante a geração de imagens sintéticas.

**Localização do Dataset:**
```
datasets/appearance_prior/cumacea_photos/
├── img_001.jpg
├── img_002.jpg
├── img_003.jpg
└── ...
```

**Classe AppearancePrior:**
```python
from campylaspis.generation import AppearancePrior

# Carregar dataset de aparência
prior = AppearancePrior("datasets/appearance_prior/cumacea_photos")

# Amostrar imagens de referência
reference_images = prior.sample_reference_images(n=3)

# Computar estatísticas de cor do dataset
color_stats = prior.compute_dataset_color_statistics()

# Amostrar da distribuição de cores
color_samples = prior.get_color_distribution_sample(n_samples=1000)
```

**Funcionalidades:**
- Carregamento automático de todas as imagens do diretório
- Computação prévia de histogramas de cor para eficiência
- Amostragem aleatória de imagens de referência
- Estatísticas agregadas de cor do dataset
- Amostragem de distribuição de cores para paletas realistas

**Teste:**
```bash
python scripts/test_appearance_prior.py
```

### Appearance Conditioning

O sistema inclui condicionamento de aparência (`AppearanceConditioner`) que extrai paletas de cores de fotografias reais e as converte em modificadores textuais para prompts de geração.

**Classe AppearanceConditioner:**
```python
from campylaspis.generation import AppearancePrior, AppearanceConditioner

# Carregar dataset de aparência
prior = AppearancePrior("datasets/appearance_prior/cumacea_photos")

# Criar condicionador de aparência
conditioner = AppearanceConditioner(prior)

# Gerar modificador de aparência
appearance_modifier = conditioner.build_appearance_prompt_modifier()
print(appearance_modifier)
# Output: "natural crustacean coloration, pale translucent body, soft marine beige tones, subtle reddish pigmentation, realistic marine specimen appearance"
```

**Funcionalidades:**
- Extração de paletas de cores dominantes usando k-means clustering
- Classificação automática de cores em categorias naturais de crustáceos
- Geração de modificadores textuais descritivos para prompts
- Amostragem múltipla do dataset para diversidade de cores
- Variação controlada de coloração para realismo individual

**Extração de Paleta:**
```python
# Extrair paleta de uma imagem específica
palette = conditioner.extract_color_palette(image, n_colors=5)
# Retorna: [(r1, g1, b1), (r2, g2, b2), ...]

# Converter paleta em modificador textual
modifier = conditioner.build_color_prompt_modifier(palette)
# Output: "pale translucent body, soft marine beige tones, subtle reddish pigmentation"
```

**Integração com Geração:**
```python
# Exemplo de uso em pipeline de geração
base_prompt = "high-resolution macro photograph of crustacean carapace"
appearance_modifier = conditioner.build_appearance_prompt_modifier()
final_prompt = f"{base_prompt}, {appearance_modifier}"

# Resultado: "high-resolution macro photograph of crustacean carapace, natural crustacean coloration, pale translucent body, soft marine beige tones, realistic marine specimen appearance"
```

**Dependências:**
```bash
pip install scikit-learn
```

**Teste:**
```bash
python scripts/test_appearance_conditioning.py
```

### Dependências para Geração

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install opencv-python Pillow
```

### Exemplo de Uso

```python
from campylaspis.generation import generate_dataset_from_taxonomy

# Caminhos para ilustrações científicas
illustration_paths = ["data/illustrations/specimen_01.png"]

# Descrições taxonômicas
descriptions = ["Carapaça oval com pseudorostrum alongado"]

# Geração com controle estrutural e condicionamento de aparência
metadata = generate_dataset_from_taxonomy(
    illustration_paths=illustration_paths,
    description_texts=descriptions,
    output_dir="generated_dataset",
    generation_mode="structure_guided",
    structure_strength=0.95,  # Alta força de condicionamento
    appearance_prior_dir="datasets/appearance_prior/cumacea_photos"  # Opcional: condicionamento de aparência
)
```

### Scripts de Teste

- `scripts/test_generation_smoke.py`: Testa o pipeline de geração
- `scripts/test_appearance_conditioning_integration.py`: Testa integração de condicionamento de aparência
- `scripts/test_appearance_metadata.py`: Testa geração de metadados com informações de condicionamento de aparência
- `example_pipeline.py`: Exemplo completo de uso

### Parâmetros Importantes

- `generation_mode`: "text_only", "structure_guided", ou "body_part"
- `structure_strength`: Força do condicionamento estrutural (0.0-1.0, recomendado 0.9+)
- `num_images_per_specimen`: Número de imagens por espécime
- `appearance_prior_dir`: Caminho opcional para dataset de fotografias reais para condicionamento de aparência

### Estrutura dos Metadados

O pipeline gera um arquivo `metadata.json` com informações completas sobre a geração:

```json
{
  "dataset_info": {
    "name": "scientific_specimen_generation_dataset",
    "description": "Generated realistic specimen images from taxonomic descriptions",
    "num_specimens": 2,
    "images_per_specimen": 4,
    "total_images": 8
  },
  "generation_config": {
    "generation_mode": "structure_guided",
    "appearance_prior_used": true,
    "appearance_prior_dataset": "datasets/appearance_prior/cumacea_photos",
    "reference_images_sampled": 3,
    "color_palette_used": [[180,165,150],[210,195,180],[140,120,100]],
    "structure_strength": 0.95,
    "model": "runwayml/stable-diffusion-v1-5",
    "controlnet": "lllyasviel/sd-controlnet-canny"
  },
  "specimens": [...]
}
```

**Campos de Condicionamento de Aparência:**
- `appearance_prior_used`: Booleano indicando se condicionamento foi usado
- `appearance_prior_dataset`: Caminho para o dataset de fotografias de referência
- `reference_images_sampled`: Número de imagens de referência amostradas
- `color_palette_used`: Paleta de cores RGB extraída das fotografias de referência

## Pipeline Completo de Classificação

┌────────────────────────────┐
│  Dataset Campylaspis       │
│  (imagens + textos)        │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ make_manifest.py           │
│ - indexa dados             │
│ - gera manifest.jsonl      │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ make_splits_by_image.py    │
│ - K-fold (estratificado)   │
│ - divisão por imagem       │
└─────────────┬──────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│            TRAIN (por fold)              │
│                                         │
│  ┌──────────────┐   ┌────────────────┐ │
│  │  Unimodal    │   │  Multimodal    │ │
│  │  (imagem)    │   │ (img + texto)  │ │
│  └──────┬───────┘   └──────┬─────────┘ │
│         │                  │            │
│         ▼                  ▼            │
│   runs/unimodal_*     runs/multimodal_* │
│   - best.pt           - best.pt         │
│   - train_log.jsonl   - train_log.jsonl │
└─────────────┬───────────────────────────┘
              │
              ▼
┌────────────────────────────┐
│ analyze_results.py         │
│ - agrega folds             │
│ - boxplot                  │
│ - Wilcoxon                 │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ Resultados finais          │
│ - Macro-F1                 │
│ - Acurácia                 │
│ - Significância estat.     │
└────────────────────────────┘

## Diagrama UNimodal
Imagem (224x224)
      │
      ▼
┌──────────────────┐
│ CNN Backbone     │  (ResNet18 / ConvNeXt)
│ (timm pretrained)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Global Avg Pool  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Classifier       │
│ (Linear)         │
└────────┬─────────┘
         │
         ▼
     Species


DIagrama MUltimodal

                  ┌──────────────────────┐
                  │ Texto (descrição)    │
                  └─────────┬────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │ DistilBERT Encoder   │
                  │ (pré-treinado)       │
                  └─────────┬────────────┘
                            │
                        Text Embedding
                            │
                            ▼
                     Linear Projection
                            │
                            ▼
                       txt_proj (D)



Imagem(s) ──► CNN Backbone ──► Pooling ──► img_emb
                                               │
                                               ▼
                                        Linear Projection
                                               │
                                               ▼
                                          img_proj (D)


                   img_proj + txt_proj
                            │
                            ▼
                    ┌────────────────┐
                    │ Gating Network │
                    │  α = sigmoid() │
                    └───────┬────────┘
                            │
              fused = α·img + (1−α)·txt
                            │
                            ▼
                    ┌────────────────┐
                    │ Classifier     │
                    │ (Linear)       │
                    └───────┬────────┘
                            │
                            ▼
                        Species
