use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[allow(dead_code)]
pub struct PeriodicTimer {
    latest: SystemTime,
    threshold_in_secs: f64,
}

#[allow(dead_code)]
impl PeriodicTimer {
    pub fn new(threshold_in_secs: f64) -> Self {
        let latest = UNIX_EPOCH;
        PeriodicTimer {
            latest,
            threshold_in_secs,
        }
    }
    pub fn maybe(&mut self, condition: bool, mut x: impl FnMut()) {
        let now = SystemTime::now();
        if condition
            || now
                .duration_since(self.latest)
                .unwrap_or(Duration::ZERO)
                .as_secs_f64()
                > self.threshold_in_secs
        {
            x();
            self.latest = now;
        }
    }
}

#[allow(dead_code)]
pub struct EchoTimer {
    start: SystemTime,
    latest: SystemTime,
    echo: bool,
}

#[allow(dead_code)]
impl EchoTimer {
    pub fn new() -> Self {
        let start = SystemTime::now();
        let echo = matches!(std::env::var("FANGS_ECHO"), Ok(x) if x == "TRUE" || x == "true");
        EchoTimer {
            start,
            latest: start,
            echo,
        }
    }

    pub fn echo(&self) -> bool {
        self.echo
    }

    pub fn stamp(&mut self, msg: &str) -> Option<String> {
        let now = SystemTime::now();
        let result = match self.echo {
            true => {
                let lapse = now
                    .duration_since(self.latest)
                    .unwrap_or(Duration::ZERO)
                    .as_secs_f64();
                let total = now
                    .duration_since(self.start)
                    .unwrap_or(Duration::ZERO)
                    .as_secs_f64();
                Some(format!(
                    "Total: {:>12.6}s, Lapse: {:>12.6}s --- {}",
                    total, lapse, msg
                ))
            }
            false => None,
        };
        self.latest = now;
        result
    }

    pub fn total_as_secs_f64(&self) -> f64 {
        let now = SystemTime::now();
        now.duration_since(self.start)
            .unwrap_or(Duration::ZERO)
            .as_secs_f64()
    }
}

#[allow(dead_code)]
pub struct TicToc {
    start: SystemTime,
    lapse: u128,
    running: bool,
}

#[allow(dead_code)]
impl TicToc {
    pub fn new() -> Self {
        Self {
            start: UNIX_EPOCH,
            lapse: 0,
            running: false,
        }
    }

    pub fn tic(&mut self) {
        if self.running {
            panic!("Clock is already running.");
        }
        self.running = true;
        self.start = SystemTime::now();
    }

    pub fn toc(&mut self) {
        let now = SystemTime::now();
        self.lapse += now
            .duration_since(self.start)
            .unwrap_or(Duration::ZERO)
            .as_nanos();
        if !self.running {
            panic!("Clock is not running.");
        }
        self.running = false;
    }

    pub fn as_secs_f64(&self) -> f64 {
        if self.running {
            panic!("Clock is running.");
        }
        (self.lapse as f64) / ((1000 * 1000 * 1000) as f64)
    }
}
