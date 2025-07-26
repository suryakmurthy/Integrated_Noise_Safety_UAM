import yaml

def conda_to_pip(yaml_file, output_file='conda_requirements.txt'):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    pip_requirements = data.get('dependencies', [])
    pip_requirements = [pkg for pkg in pip_requirements if isinstance(pkg, str) or 'pip' in pkg]

    if any(isinstance(req, dict) for req in pip_requirements):
        pip_requirements = [
            req
            for deps in pip_requirements
            if isinstance(deps, dict) and 'pip' in deps
            for req in deps['pip']
        ]

    with open(output_file, 'w') as f:
        for requirement in pip_requirements:
            f.write(f"{requirement}\n")

# Usage
# conda_to_pip('environment.yml')
import torch

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Testing GPU access...")
        device = torch.device("cuda")
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device=device)
            print("Successfully moved tensor to GPU:")
            print(x)
        except Exception as e:
            print(f"Failed to move tensor to GPU: {e}")
    else:
        print("CUDA is not available. Using CPU...")
        device = torch.device("cpu")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print("Tensor on CPU:")
        print(x)

test_cuda()
