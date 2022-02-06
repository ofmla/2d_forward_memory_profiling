"""
inversion script
"""
from shot_control_inversion import ControlInversion

def main():
    control_inv = ControlInversion()
    control_inv.run_inversion()

if __name__ == "__main__":
    main()
