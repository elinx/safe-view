import subprocess
import sys

def main():
    try:
        # Run the installed script with --help
        result = subprocess.run(["safe-view", "--help"], capture_output=True, text=True, check=True)
        print("safe-view --help output:")
        print(result.stdout)
        if "usage: safe-view" not in result.stdout:
            print("Error: 'usage: safe-view' not found in help output.")
            sys.exit(1)
        print("Smoke test passed!")
    except FileNotFoundError:
        print("Error: 'safe-view' command not found. Is the package installed correctly?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running 'safe-view --help': {e}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
