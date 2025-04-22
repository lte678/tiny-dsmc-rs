use std::cell::RefCell;
use std::f64::consts::PI;
use std::iter::zip;
use std::fs;
use std::io;

use yaml_rust::YamlLoader;
use rand::prelude::*;

const BOLTZ: f64 = 1.3806e-23;


fn main() {
    println!("> Tiny-DSMC-RS");
    println!("> By Leon Teichroeb\n");

    let _ = fs::remove_file("output.log");
    let yaml = &YamlLoader::load_from_str(
        &fs::read_to_string("sim.yaml").unwrap()
    ).unwrap()[0];
    println!("Loaded sim.yaml.");

    let xmin = yaml["xmin"].as_f64().unwrap(); // "XF"
    let xmax = yaml["xmax"].as_f64().unwrap(); // "XR"
    let cell_count = yaml["cell-count"].as_i64().unwrap() as usize; // "MNC"
    let subcells_per_cell = yaml["subcells-per-cell"].as_i64().unwrap() as usize; // "NSC"

    let dt = yaml["timestep"].as_f64().unwrap(); // "DTM"
    let steps_per_sample = yaml["steps-per-sample"].as_i64().unwrap() as usize; // "NIS"
    // Interval in nr. of samples to take before saving the output.
    let save_interval = yaml["save-interval"].as_i64().unwrap() as usize; // "NSP"
    let save_count = yaml["save-count"].as_i64().unwrap() as usize; // "NPT"

    let molecular_diameter = yaml["molecular-diameter"].as_f64().unwrap(); // "SP(2)"

    let params = SimParameters {
        rng: RefCell::new(rand::rng()),
        density: yaml["density"].as_f64().unwrap(),
        temperature: yaml["temperature"].as_f64().unwrap(),
        molecular_mass: yaml["molecular-mass"].as_f64().unwrap(),
        p_per_p: yaml["particles-per-particle"].as_f64().unwrap(),
    };

    let coll_crosssection = PI * molecular_diameter * molecular_diameter;

    let geom = Geometry::new(xmin, xmax, cell_count, subcells_per_cell);
    let mut cell_data: Vec<CellData> = Vec::with_capacity(geom.cells.len());
    let mut subcell_data: Vec<SubcellData> = Vec::with_capacity(geom.cells.len() * geom.subcells_per_cell);
    for _ in 0..geom.cells.len() {
        let maximal_coll_vol = coll_crosssection * 300.0 * (params.temperature/300.0).sqrt();
        let selection_remainder = uniform(&params);
        cell_data.push(
            CellData { maximal_coll_vol, selection_remainder, molecule_index: 0, molecule_count: 0,
            u_mean: 0.0, v_mean: 0.0, w_mean: 0.0, kin_energy_mean: 0.0 }
        )
    }
    for _ in 0..(geom.cells.len() * geom.subcells_per_cell) {
        subcell_data.push(
            SubcellData { molecule_index: 0, molecule_count: 0 }
        )
    }

    let mut molecules: Vec<Molecule> = init_molecules(
        &params,
        &geom,        
    );
    // Cross-reference table to find molecules in a given cell
    let mut cell_molecules: Vec<usize> = vec![0; molecules.len()]; // "IR"
    println!("Simulation contains {} molecules.", molecules.len());



    let mut time: f64 = 0.0;
    for _ in 1..=save_count {
        for _ in 1..=save_interval {
            for _ in 1..=steps_per_sample {
                time += dt;
                move_molecules(&mut molecules, &geom, dt);
                index_molecules(&mut cell_molecules, &mut cell_data, &mut subcell_data, &geom, &mut molecules);
            }
            sample_molecules(&mut cell_data, &molecules, &cell_molecules, &params);
        }
        let mut f = fs::OpenOptions::new().append(true).create(true).open("output.log").unwrap();
        output_cell_data(&mut f, &cell_data, &geom, &params, time).unwrap();
        println!("Time = {:.6}s", &time);
    }
}


struct SimParameters<R: Rng> {
    rng: RefCell<R>, // The random number generator to use for the sim state. Has interior mutability.
    density: f64, // "FND"
    temperature: f64, // "FTMP"
    molecular_mass: f64, // "SP(1)"
    /// Number of real particles represented by one simulation molecule
    p_per_p: f64, // "FNUM"
}


struct Cell {
    xmin: f64, // "CG(1, ...)"
    xmax: f64, // "CG(2, ...)"
    cell_size: f64, // "CG(3, ...)" and "CC"
}


struct Geometry {
    xmin: f64, // "XF"
    xmax: f64, // "XR"
    cell_size: f64, // "CW"
    // Length == "MNC"
    cells: Vec<Cell>,
    subcells_per_cell: usize, // "NSC"
}


impl Geometry {
    fn new(xmin: f64, xmax: f64, cell_count: usize, subcells_per_cell: usize) -> Geometry {
        let cell_size = (xmax - xmin) / cell_count as f64;
        let mut cells = Vec::with_capacity(cell_count);
        let mut x = xmin;
        for _ in 0..cell_count {
            cells.push(
                Cell {
                    xmin: x, xmax: x + cell_size, cell_size
                }
            );
            x += cell_size;
        }
        Geometry { xmin, xmax, cell_size, cells, subcells_per_cell }
    }

    /// Returns the cell index of the given point
    fn cell_index(&self, x: f64) -> usize {
       let i = ((x - self.xmin) / self.cell_size).floor() as i64;
       i.clamp(0, (self.cells.len() - 1) as i64) as usize
    }

    /// Returns the subcell index of the given point
    fn subcell_index(&self, x: f64, cell_index: Option<usize>) -> usize {
        let cell_i = cell_index.unwrap_or(self.cell_index(x));
        let subcell = ((self.subcells_per_cell as f64) * (x - self.cells[cell_i].xmin) / self.cell_size).floor() as i64;
        subcell.clamp(0, self.subcells_per_cell as i64) as usize + cell_i * self.subcells_per_cell
    }


    /// Returns the cell index of the given subcell. "ISC"
    fn subcell_to_cell(&self, subcell: usize) -> usize {
        subcell / self.subcells_per_cell
    }
}


struct Molecule {
    x: f64, // "PP"
    // These constitute the array "PV"
    u: f64,
    v: f64,
    w: f64,
    // The subcell that the molecule belongs to.
    subcell: usize, // "IP"
}


struct SubcellData {
    molecule_index: usize, // "ISCG(1, ...)"
    molecule_count: usize, // "ISCG(2, ...)"       
}

struct CellData {
    // (relative speed * coll. cross section)
    maximal_coll_vol: f64, // "CCG(1, ...)"
    // Remainder of particles when selecting a non-integral number.
    selection_remainder: f64, // "CCG(2, ...)"
    // Points to start of molecule indices in the cell_molecules data structure
    molecule_index: usize, // "IC(1, ...)"
    molecule_count: usize, // "IC(2, ...)"
    u_mean: f64,
    v_mean: f64,
    w_mean: f64,
    kin_energy_mean: f64,
}

fn init_molecules<R: Rng>(params: &SimParameters<R>, geom: &Geometry) -> Vec<Molecule> {
    // Calculate the expected number of simulation particles based on the domain size and density.
    let molecule_count = (params.density * (geom.xmax - geom.xmin) / params.p_per_p).ceil() as usize; // "NM"
    let mut molecules = Vec::with_capacity(molecule_count);
    
    let c_likely = (2.0 * BOLTZ * params.temperature / params.molecular_mass).sqrt();
    // We can only add integral amounts of molecules, but we want to carry the fractional part.
    let mut m_count_remainder = 0.0;
    for (cell_i, cell) in geom.cells.iter().enumerate() {
        let m_count = params.density * cell.cell_size / params.p_per_p + m_count_remainder;
        m_count_remainder = m_count.fract();
        for _ in 0..m_count.floor() as usize {
            let x = cell.xmin + uniform(params) * cell.cell_size;
            molecules.push(
                Molecule {
                    x,
                    u: sample_c_therm(params, c_likely),
                    v: sample_c_therm(params, c_likely),
                    w: sample_c_therm(params, c_likely),
                    subcell: geom.subcell_index(x, Some(cell_i))
                }
            )
        }
    }
    // Not actually important, but it should match anyway
    assert_eq!(molecules.len(), molecule_count);
    return molecules;
}


fn sample_molecules<R: Rng>(cell_data: &mut Vec<CellData>, molecules: &Vec<Molecule>, cell_molecules: &Vec<usize>, params: &SimParameters<R>) {
    for cd in cell_data.iter_mut() {
        cd.u_mean = 0.0;
        cd.v_mean = 0.0;
        cd.w_mean = 0.0;
        cd.kin_energy_mean = 0.0;
    }
    
    for cd in cell_data.iter_mut() {
        for i in cd.molecule_index..(cd.molecule_index + cd.molecule_count) {
            let mol_i = cell_molecules[i];
            let m = &molecules[mol_i];
            cd.u_mean += m.u;
            cd.v_mean += m.v;
            cd.w_mean += m.w;
            cd.kin_energy_mean += 0.5*params.molecular_mass*(m.u*m.u + m.v*m.v + m.w*m.w);
        }
    }

    for cd in cell_data.iter_mut() {
        cd.u_mean /= cd.molecule_count as f64 + 1e-10;
        cd.v_mean /= cd.molecule_count as f64 + 1e-10;
        cd.w_mean /= cd.molecule_count as f64 + 1e-10;
        cd.kin_energy_mean /= cd.molecule_count as f64 + 1e-10;
    }
}


fn move_molecules(molecules: &mut Vec<Molecule>, geom: &Geometry, dt: f64) {
    for m in molecules.iter_mut() {
        let x = m.x + m.u * dt;
        if x < geom.xmin {
            m.x = 2.0*geom.xmin - x;
            m.u = -m.u;
        } else if x > geom.xmax {
            m.x = 2.0*geom.xmax - x;
            m.u = -m.u;
        } else {
            m.x = x;
        }
    }
}


fn index_molecules(cell_molecules: &mut Vec<usize>, cell_data: &mut Vec<CellData>, subcell_data: &mut Vec<SubcellData>, geom: &Geometry, molecules: &mut Vec<Molecule>) {
    for cd in cell_data.iter_mut() {
        cd.molecule_count = 0;
    }
    for scd in subcell_data.iter_mut() {
        scd.molecule_count = 0;
    }

    // Count molecules per cell
    for m in molecules.iter_mut() {
        let cell_index = geom.cell_index(m.x);
        let subcell_index = geom.subcell_index(m.x, Some(cell_index));
        // Update the subcell of molecules that may have moved out of their cell during the move step.
        // This has been moved out of the MOVE0S routine of the original program.
        m.subcell = subcell_index;
        subcell_data[subcell_index].molecule_count += 1;
        cell_data[cell_index].molecule_count += 1;
    }
    // Assign the start index of each cell
    let mut mol_index = 0;
    for cd in cell_data.iter_mut() {
        cd.molecule_index = mol_index;
        mol_index += cd.molecule_count
    }
    // Assign the start index of each subcell
    let mut mol_index = 0;
    for scd in subcell_data.iter_mut() {
        scd.molecule_index = mol_index;
        mol_index += scd.molecule_count;
        // Set molecule count to zero, since it will be reused to construct the subcell molecules array. 
        scd.molecule_count = 0;
    }
    // Populate the cell to molecule index cross ref table
    for (m_i, m) in molecules.iter().enumerate() {
        let sc_i = geom.subcell_index(m.x, None);
        let k = subcell_data[sc_i].molecule_index + subcell_data[sc_i].molecule_count;
        cell_molecules[k] = m_i;
        subcell_data[sc_i].molecule_count += 1;
    }
}


fn output_cell_data<R: Rng>(f: &mut impl io::Write, cell_data: &Vec<CellData>, geom: &Geometry, params: &SimParameters<R>, time: f64) -> Result<(), io::Error> {
    writeln!(f, "TIME {:.4e}", time)?;
    writeln!(f, "FLOWFIELD PROPERTIES")?;
    writeln!(f, "cell, x, u, v, w, temp")?;
    for (cell_i, (cell, cd)) in zip(&geom.cells, cell_data).enumerate() {
        let macro_kin_energy = 0.5*params.molecular_mass*(cd.u_mean*cd.u_mean + cd.v_mean*cd.v_mean + cd.w_mean*cd.w_mean);
        let temperature = (cd.kin_energy_mean - macro_kin_energy) / (1.5 * BOLTZ);
        let x_center = 0.5*(cell.xmin + cell.xmax);
        writeln!(
            f,
            "{}, {:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}",
            cell_i,
            x_center,
            cd.u_mean,
            cd.v_mean,
            cd.w_mean,
            temperature
        )?;
    }
    writeln!(f, "")?;
    Ok(())
}


/// Sample the thermal velocity
fn sample_c_therm<R: Rng>(params: &SimParameters<R>, c_likely: f64) -> f64 {
    let a = (-uniform(params).ln()).sqrt();
    let b = 6.283185308 * uniform(params);
    a * b.sin() * c_likely
}

/// Sample from uniform distribution between 0.0 and 1.0
fn uniform<R: Rng>(params: &SimParameters<R>) -> f64 {
    params.rng.borrow_mut().random()
}