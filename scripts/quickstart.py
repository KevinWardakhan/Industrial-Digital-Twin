import argparse
from pathlib import Path
from digital_twin.simple_digital_twin import SimpleDigitalTwin

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to consolidated Parquet data')
    p.add_argument('--config', required=True, help='Path to DHC config JSON')
    p.add_argument('--equipment', default='F101', help='Equipment name')
    p.add_argument('--n', type=int, default=1000, help='Number of samples')
    args = p.parse_args()

    dt = SimpleDigitalTwin(csv_path=args.data, config_path=args.config)
    dt.analyze_single_equipment_only(args.equipment)
    # dt.generate_virtual_for_single_equipment(args.equipment, n_samples=args.n)

if __name__ == '__main__':
    main()\n
