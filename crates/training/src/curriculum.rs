//! Curriculum schedule for dream sizes. See `decisions/05-curriculum.md`.

#[derive(Clone, Debug)]
pub struct CurriculumStage {
    /// Inclusive iteration index this stage runs through.
    pub through_iter: usize,
    /// Maximum dream-program size for this stage.
    pub max_size: u32,
}

#[derive(Clone, Debug)]
pub struct Curriculum {
    pub stages: Vec<CurriculumStage>,
}

impl Default for Curriculum {
    fn default() -> Self {
        Self {
            stages: vec![
                CurriculumStage { through_iter: 2,   max_size: 3  },
                CurriculumStage { through_iter: 5,   max_size: 5  },
                CurriculumStage { through_iter: 10,  max_size: 7  },
                CurriculumStage { through_iter: 15,  max_size: 9  },
                CurriculumStage { through_iter: 25,  max_size: 11 },
                CurriculumStage { through_iter: 50,  max_size: 13 },
                CurriculumStage { through_iter: 999, max_size: 15 },
            ],
        }
    }
}

impl Curriculum {
    pub fn max_size_for(&self, iteration: usize) -> u32 {
        for s in &self.stages {
            if iteration <= s.through_iter { return s.max_size; }
        }
        self.stages.last().map(|s| s.max_size).unwrap_or(7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps() {
        let c = Curriculum::default();
        assert_eq!(c.max_size_for(0), 3);
        assert_eq!(c.max_size_for(2), 3);
        assert_eq!(c.max_size_for(3), 5);
        assert_eq!(c.max_size_for(11), 9);
        assert_eq!(c.max_size_for(20), 11);
        assert_eq!(c.max_size_for(50), 13);
        assert_eq!(c.max_size_for(1000), 15);
    }
}
