# MAIA4402 - Demo | Despliegue de Modelos de Machine Learning en AWS

Este es un Demo para desplegar modelos ya entrenados en AWS usando Infraestructura como Código (IaC) con AWS CDK. Y a la par, desplegar un servicio en streamlit para interactuar con los modelos. con una interfaz amigable.

En general, el procedimiento un despliegue de este estilo es el siguiente:

1. Crear un bucket S3 para almacenar los artefactos del modelo. (el archivo .tar.gz)
2. Crear un modelo de Sagemaker con el artefacto del modelo.
3. Crear un endpoint de Sagemaker para el modelo.
   1. Este paso puede costar bastante si se configura mal: Revisen el free tier de sagemaker en [La Página de pricing de SageMaker](https://aws.amazon.com/sagemaker-ai/pricing/) y elijan las instancias adecuadas o serverless.
4. Crear la tarea de Fargate para la aplicación Streamlit.
5. Crear un servicio de ECS Fargate para desplegar la aplicación Streamlit.
6. Crear un balanceador de carga para el servicio de ECS Fargate. (opcional)

Alternativamente, puede seguir [este tutorial de apps de streamlit en notebooks de sagemaker](https://aws.amazon.com/blogs/machine-learning/build-streamlit-apps-in-amazon-sagemaker-studio/) para desplegar la aplicación en un notebook de SageMaker.

## Estructura del Repositorio

```bash
.
├── app.py                  # Punto de entrada principal de la aplicación CDK
├── BaseModel.py           # Definición de la clase base para metadatos de modelos de ML
├── config.json            # Archivo de configuración para definiciones y ajustes de modelos
├── models/               # Directorio que contiene implementaciones de modelos
│   └── pytorch_yolo/    # Implementación del modelo PyTorch YOLO
│       └── code/        # script de inferencia del modelo
├── stacks/              # Definiciones de stacks AWS CDK para desplegar en CloudFormation
│   ├── model_deployment_base_stack.py    # Stack de infraestructura base, buckets y roles
│   ├── model_deployment_stack.py         # Stack de despliegue de modelos, endpoints y tarjetas
│   └── streamlit_stack.py                # Stack de despliegue de UI Streamlit, ECS y balanceador de carga
└── streamlit_app/       # Aplicación web Streamlit
    ├── Dockerfile       # Definición del contenedor para la app Streamlit
    ├── streamlit_app.py # Código principal de la aplicación Streamlit
    └── utils.py        # Funciones de utilidad para interacciones con los modelos
```

## Instrucciones de Uso

### Prerrequisitos

- Python 3.13+
- AWS CLI configurado con credenciales apropiadas
- Docker instalado para desarrollo local
- AWS CDK CLI instalado (`npm install -g aws-cdk`)
- Una cuenta AWS con permisos para crear:
  - Endpoints de SageMaker
  - Servicios ECS Fargate
  - Buckets S3
  - Roles IAM -> en caso de fallar en los labs, puede usar el LabRole
  - Logs de CloudWatch

### Instalación

1. Clonar el repositorio e instalar dependencias:

```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate.bat

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Configurar credenciales AWS:

```bash
aws configure
```

3. Desplegar la infraestructura:

```bash
cdk deploy --all # Desplegar todos los stacks

# O desplegar stacks específicos
cdk deploy MAIA4402-BaseStack # para el modelo
cdk deploy StreamlitStack # para la interfaz Streamlit

```

### Inicio Rápido

1. Configurar los modelos en `config.json`:

```json
{
  "models": [
    {
      "name": "YOLOPersonDetection",
      "framework": "pytorch",
      "data_path": "./data/pytorch_yolo/",
      "path": "./models/pytorch_yolo/",
      "file_name": "model.pt",
      "image_uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker",
      "problem_type": "object detection",
      "serverless": true
    }
  ]
}
```

2. Desplegar el modelo en SageMaker:

```bash
cdk deploy MAIA4402-BaseStack
```

3. Desplegar la interfaz Streamlit (opcional, puedes ejecutarlo localmente)

```bash
cdk deploy StreamlitStack
```

## Arquitectura de la Solución

![Diagrama de arquitectura](./docs/Arquitectura.jpg)

### Buckets S3

- `cdk-sagemaker-data-{account}-{region}`: Almacena datos de muestra, para pruebas de modelos
- `cdk-sagemaker-models-{account}-{region}`: Almacena artefactos de modelos

### Recursos SageMaker

- Endpoints de Modelo: Endpoints serverless de inferencia para cada modelo desplegado
- Tarjetas de Modelo: Metadatos y documentación para modelos desplegados

### Recursos ECS

- Servicio Fargate: Ejecuta la aplicación Streamlit en un contenedores serverless
- Tarea Fargate: Define la configuración del contenedor y la red
- Balanceador de Carga de Aplicación: Dirige el tráfico a la aplicación Streamlit, se tiene con el objetivo de tener un dominio y no depender de la IP pública del contenedor
- Grupo de Seguridad: Controla el acceso a la aplicación Streamlit
- VPC: Proporciona aislamiento de red para la aplicación Streamlit y para acceder a otros servicios de AWS

### Roles IAM

- Rol de Ejecución SageMaker: Permisos para despliegue e inferencia de modelos
- Rol de Tarea ECS: Permisos para que la aplicación Streamlit acceda a servicios AWS

## Despliegue

1. Prerrequisitos:

- AWS CDK CLI instalado
- Credenciales AWS configuradas
- Docker instalado

2. Pasos de Despliegue:

```bash
# Bootstrap CDK (solo primera vez)
cdk bootstrap

# Desplegar infraestructura base
cdk deploy MAIA4402-BaseStack

# Desplegar interfaz Streamlit
cdk deploy StreamlitStack
```

3. Configuración del Entorno:

```bash
# Establecer variables de entorno
export CDK_DEFAULT_ACCOUNT=<your-account-id>
export CDK_DEFAULT_REGION=<your-region>
```

4. Monitoreo:

- Logs en CloudWatch: `/aws/sagemaker/Endpoints` y `/ecs/StreamlitContainer`
- Métricas CloudWatch: Invocaciones y latencia del endpoint SageMaker
