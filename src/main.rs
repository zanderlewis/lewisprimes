use num_bigint::{BigUint, RandBigInt, ToBigUint};
use num_traits::{One, Zero};
use rand::thread_rng;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use threadpool::ThreadPool;

// Function to perform the Miller-Rabin primality test
fn miller_rabin(n: &BigUint, k: u32) -> bool {
    if n == &2.to_biguint().unwrap() || n == &3.to_biguint().unwrap() {
        return true;
    }
    if n < &2.to_biguint().unwrap() || n % 2.to_biguint().unwrap() == Zero::zero() {
        return false;
    }

    let mut d = n - 1.to_biguint().unwrap();
    let mut r = 0;
    while &d % 2.to_biguint().unwrap() == Zero::zero() {
        d /= 2.to_biguint().unwrap();
        r += 1;
    }

    'witness_loop: for _ in 0..k {
        let a = thread_rng()
            .gen_biguint_range(&2.to_biguint().unwrap(), &(n - 2.to_biguint().unwrap()));
        let mut x = a.modpow(&d, n);
        if x == One::one() || x == n - 1.to_biguint().unwrap() {
            continue;
        }
        for _ in 0..r - 1 {
            x = x.modpow(&2.to_biguint().unwrap(), n);
            if x == n - 1.to_biguint().unwrap() {
                continue 'witness_loop;
            }
        }
        return false;
    }
    true
}

// Function to check for Lewis Primes: 10^n - 11
fn find_lewis_prime(n: u32) -> Option<BigUint> {
    let ten = 10.to_biguint().unwrap();
    let candidate = ten.pow(n) - 11.to_biguint().unwrap();
    // 40 iterations for a good balance between speed and accuracy
    if miller_rabin(&candidate, 40) {
        Some(candidate)
    } else {
        None
    }
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let num_threads = 4; // Adjust the number of threads as needed
    let lower_limit = 5_001; // Set the lower limit
    let upper_limit = 10_000; // Set the upper limit

    // Delete the file if it exists
    let path = "primes.txt";
    if Path::new(path).exists() {
        std::fs::remove_file(path)?;
    }

    // Append so_far.txt to primes.txt
    let so_far = "so_far.txt";
    if Path::new(so_far).exists() {
        std::fs::copy(so_far, path)?;
    }

    // Open the file in append mode
    let file = Arc::new(Mutex::new(
        OpenOptions::new().create(true).append(true).open(path)?,
    ));

    // Create a thread pool
    let pool = ThreadPool::new(num_threads);

    for n in lower_limit..=upper_limit {
        let file = Arc::clone(&file);
        pool.execute(move || {
            if let Some(prime) = find_lewis_prime(n) {
                println!("Lewis Prime for n = {}: {}", n, prime);
                let mut file = file.lock().unwrap();
                writeln!(file, "{}", n).unwrap();
            } else {
                println!("No Lewis Prime for n = {}", n);
            }
        });
    }

    // Wait for all threads to finish
    pool.join();

    let duration = start.elapsed();
    println!("Execution time: {:?}", duration);

    // Delete so_far.txt and rename primes.txt to so_far.txt
    if Path::new(so_far).exists() {
        std::fs::remove_file(so_far)?;
    }
    std::fs::rename(path, so_far)?;

    Ok(())
}
