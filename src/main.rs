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
    let coll_crosssection = PI * molecular_diameter * molecular_diameter; // "SPM(1)"
    let viscosity_powlaw = yaml["viscosity-powlaw"].as_f64().unwrap(); // "SPM(3)"

    let params = SimParameters {
        rng: RefCell::new(rand::rng()),
        density: yaml["density"].as_f64().unwrap(),
        temperature: yaml["temperature"].as_f64().unwrap(),
        molecular_mass: yaml["molecular-mass"].as_f64().unwrap(),
        coll_crosssection: coll_crosssection,
        reference_temp: yaml["ref-temperature"].as_f64().unwrap(),
        viscosity_powlaw,
        vss_scattering: yaml["vss-scattering"].as_f64().unwrap(),
        gamma_fac: gamma(5.0/2.0 - viscosity_powlaw),
        p_per_p: yaml["particles-per-particle"].as_f64().unwrap(),
    };


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
    // Output for t = 0.0
    index_molecules(&mut cell_molecules, &mut cell_data, &mut subcell_data, &geom, &mut molecules);
    sample_molecules(&mut cell_data, &molecules, &cell_molecules, &params);
    let mut f = fs::OpenOptions::new().append(true).create(true).open("output.log").unwrap();
    output_cell_data(&mut f, &cell_data, &geom, &params, time).unwrap();
    
    for _ in 1..=save_count {
        for _ in 1..=save_interval {
            for _ in 1..=steps_per_sample {
                time += dt;
                move_molecules(&mut molecules, &geom, dt);
                index_molecules(&mut cell_molecules, &mut cell_data, &mut subcell_data, &geom, &mut molecules);
                collide_molecules(&mut molecules, &geom, &mut cell_data, &subcell_data, &cell_molecules, &params, dt);
            }
            sample_molecules(&mut cell_data, &molecules, &cell_molecules, &params);
        }
        let mut f = fs::OpenOptions::new().append(true).create(true).open("output.log").unwrap();
        output_cell_data(&mut f, &cell_data, &geom, &params, time).unwrap();
        debug_molecules(&molecules, &geom, &subcell_data, &cell_molecules);
        println!("Time = {:.6}s", &time);
    }
}


struct SimParameters<R: Rng> {
    rng: RefCell<R>, // The random number generator to use for the sim state. Has interior mutability.
    density: f64, // "FND"
    temperature: f64, // "FTMP"
    molecular_mass: f64, // "SP(1)"
    coll_crosssection: f64, // "SPM(1)"
    reference_temp: f64, // "SPM(2)"
    viscosity_powlaw: f64, // "SPM(3)"
    vss_scattering: f64, // "SPM(4)"
    // Cached result of gamma(5/2 - viscosity_powlaw)
    gamma_fac: f64, // "SPM(5)"
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
    println!("Thermal velocity = {:.3e}m/s", c_likely);
    // We can only add integral amounts of molecules, but we want to carry the fractional part.
    let mut m_count_remainder = 0.0;
    for (cell_i, cell) in geom.cells.iter().enumerate() {
        if cell_i >= geom.cells.len() / 2 {
            continue;
        }
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
    // assert_eq!(molecules.len(), molecule_count);
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


fn collide_molecules<R: Rng>(molecules: &mut Vec<Molecule>, geom: &Geometry, cell_data: &mut Vec<CellData>, subcell_data: &Vec<SubcellData>, cell_molecules: &Vec<usize>, params: &SimParameters<R>, dt: f64) {
    
    for (cell_i, (c, cd)) in zip(&geom.cells, cell_data).enumerate() {
        // Keep track of the maximum collision volume. We will update this as we go along.
        let mut max_coll_volume = cd.maximal_coll_vol;
        
        let n = cd.molecule_count as f64;
        let n_sel = 0.5 * n * n * params.p_per_p * max_coll_volume * dt / c.cell_size + cd.selection_remainder;
        cd.selection_remainder = n_sel.fract();
        let n_sel = n_sel as usize;
        if n_sel < 2 {
            cd.selection_remainder += n_sel as f64;
            return;
        }
        for _ in 0..n_sel {
            // Select a random molecule from the given cell
            let i_cell = cd.molecule_index + params.rng.borrow_mut().random_range(0..cd.molecule_count);
            let i = cell_molecules[i_cell];
            let m_i = &mut molecules[i];
            let subcell_i = geom.subcell_index(m_i.x, Some(cell_i));
            // TODO: There is an algorithm to deal with this.
            assert!(subcell_data[subcell_i].molecule_count > 1);
            // Select a random second molecule to collide with.
            let j_cell = cd.molecule_index + sample_without(params, cd.molecule_count, i_cell - cd.molecule_index);
            let j = cell_molecules[j_cell];
            // Special functions to get two mutable references to the molecules at once, since we know they
            // will not overlap. Otherwise, if you do not want to use a Rust 1.86 feature, just borrow mutably one
            // at a time when setting the post collision velocity.
            let [m_i, m_j] = molecules.get_disjoint_mut([i, j]).unwrap();

            let u_rel = m_i.u - m_j.u; // "VRC"
            let v_rel = m_i.v - m_j.v;
            let w_rel = m_i.w - m_j.w;
            let c_rel2 = u_rel*u_rel + v_rel*v_rel + w_rel*w_rel;
            let c_rel = c_rel2.sqrt();
            
            let coll_volume = c_rel * params.coll_crosssection * 
                ((2.0*BOLTZ*params.reference_temp)/(0.5*params.molecular_mass*c_rel2))
                .powf(params.viscosity_powlaw - 0.5) / params.gamma_fac;
            
            max_coll_volume = max_coll_volume.max(coll_volume);

            if uniform(params) < coll_volume / cd.maximal_coll_vol {
                // The collision has been accepted.
                // Basically, it seems that we sample collisions to achieve the rate predicted by maximal_coll_vol, and then
                // thin it out (hopefully only by a small amount) to achieve the actual target collision rate.
                // Center of Mass velocity (CoM)
                let u_com = 0.5*(m_i.u + m_j.u); // "VCCM"
                let v_com = 0.5*(m_i.v + m_j.v); 
                let w_com = 0.5*(m_i.w + m_j.w); 
                // Use fixed VHS logic for now
                assert!((params.vss_scattering - 1.0).abs() < 1e-3);
                // Elevation angle
                let cos_theta = 2.0*uniform(params) - 1.0;
                // Azimuthal angle
                let phi = 2.0*PI*uniform(params);
                let a = (1.0 - cos_theta*cos_theta).sqrt();
                // Determine post-collision relative velocity
                let u_rel_post = cos_theta * c_rel; // "VRCP"
                let v_rel_post = a * phi.cos() * c_rel;
                let w_rel_post = a * phi.sin() * c_rel;
                
                // Set post-collision velocity.
                m_i.u = u_com + 0.5*u_rel_post;
                m_i.v = v_com + 0.5*v_rel_post;
                m_i.w = w_com + 0.5*w_rel_post;
                
                m_j.u = u_com - 0.5*u_rel_post;
                m_j.v = v_com - 0.5*v_rel_post;
                m_j.w = w_com - 0.5*w_rel_post;
            }
        }
        // Update our estimate for the maximum collision volume.
        cd.maximal_coll_vol = max_coll_volume;
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


/// This function asserts some invariants that have to do with the molecules.
/// The cross-referencing data structures that need to be painstakingly kept up to date
/// are of particular interest.
fn debug_molecules(molecules: &Vec<Molecule>, geom: &Geometry, subcell_data: &Vec<SubcellData>, cell_molecules: &Vec<usize>) {
    for m in molecules {
        assert_eq!(m.subcell, geom.subcell_index(m.x, None))
    }
    for (subcell_i, scd) in subcell_data.iter().enumerate() {
        let mol_i = scd.molecule_index;
        for m_i in &cell_molecules[mol_i..(mol_i+scd.molecule_count)] {
            assert_eq!(molecules[*m_i].subcell, subcell_i)
        }
    }
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

/// Samples from numbers between 0 and the upper bound (exclusive) while excluding the
/// excludee.
fn sample_without<R: Rng>(params: &SimParameters<R>, upper_bound: usize, excludee: usize) -> usize {
    let mut candidate = params.rng.borrow_mut().random_range(0..upper_bound-1);
    if candidate == excludee {
        candidate = upper_bound - 1;
    }
    candidate
}


/// The mathematical "Gamma"-Function
fn gamma(x: f64) -> f64 {
    let mut a: f64 = 1.0;
    let mut x = x;
    if x < 1.0 {
        a /= x;
    } else {
        x -= 1.0;
        while x > 1.0 {
            a *= x;
            x -= 1.0;
        }
    }
    a*(1.0 - 0.5748646*x + 0.9512363*x.powf(2.0) - 0.6998588*x.powf(3.0) + 0.4245549*x.powf(4.0) - 0.1010678*x.powf(5.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        // From https://en.wikipedia.org/wiki/Gamma_function
        assert!((gamma(1.5) - 0.5*PI.sqrt()).abs() < 1e-3);
        assert!((gamma(1.0) - 1.0).abs() < 1e-3);
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-3);
    }
}