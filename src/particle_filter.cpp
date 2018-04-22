/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	// Create the normal distributions for x, y, and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize particles based on the normal distributions from initial coordinates
	for (int i = 0; i < num_particles; i++) {
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Create normal distributions
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// Loop over every particle
	for (int i = 0; i < num_particles; i++) {

		// Calculate new state dependent on yaw_rate
		if (fabs(yaw_rate) < EPS)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add noise to particles
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {

		// Initialize minimum distance to the maximum value possible
		double min_dist = numeric_limits<double>::max();

		// Initialize landmark id to impossible value
		int map_id = -1;

		// Get current landmark
		LandmarkObs obs = observations[i];

		for (int j = 0; j < predicted.size(); j++) {
			// Get predicted landmark
			LandmarkObs pred = predicted[j];

			// Calculate distance between observed and predicted landmarks
			double distance = dist(obs.x, obs.y, pred.x, pred.y);

			// Updated the nearest neighbor
			if(distance < min_dist)
			{
				min_dist = distance;
				map_id = pred.id;
			}
		}
		// Bind closest predicted landmark to observation
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Parts of the weight equation that are constant
	const double denominator_overall = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	const double denominator_x = 2 * std_landmark[0] * std_landmark[0];
	const double denominator_y = 2 * std_landmark[1] * std_landmark[1];

	// Loop over every particle
	for (int i = 0; i < num_particles; i++) {

		// Get particle state
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// Landmarks within the range of the sensor
		vector<LandmarkObs> predicted_landmarks;

		// Loop over every landmark
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			double dx = p_x - landmark_x;
			double dy = p_y - landmark_y;

			// Distance between particle and landmark
			double distance = sqrt(dx * dx + dy * dy);

			// Only landmarks within the sensor range are of interest
			if (fabs(distance) <= sensor_range)
			{
				predicted_landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
			}
		}

		// Transform observation coordinates to map coordinate system
		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); j++) {
			double map_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
			double map_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
			transformed_observations.push_back(LandmarkObs{observations[j].id, map_x, map_y});
		}

		// Associate observations to landmarks
		dataAssociation(predicted_landmarks, transformed_observations);

		// Reset weight
		particles[i].weight = 1.0;

		for (int j = 0; j < transformed_observations.size(); j++) {
			double o_x = transformed_observations[j].x;
			double o_y = transformed_observations[j].y;
			int landmark_id = transformed_observations[j].id;

			double pred_x, pred_y;

			// Get prediction associated with this observation
			for (int k = 0; k < predicted_landmarks.size(); k++) {
				if (predicted_landmarks[k].id == landmark_id)
				{
					pred_x = predicted_landmarks[k].x;
					pred_y = predicted_landmarks[k].y;
					break;
				}
			}

			// Calculate the particle weight
			double dist_x = o_x - pred_x;
			double dist_y = o_y - pred_y;
			particles[i].weight *= denominator_overall * exp(-((dist_x * dist_x)/denominator_x + (dist_y * dist_y)/denominator_y));
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<double> weights;
	double max_weight = numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if (particles[i].weight > max_weight)
		{
			max_weight = particles[i].weight;
		}
	}

	// Get random starting index for resampling
	uniform_int_distribution<int> dist_int(0, num_particles - 1);
	int index = dist_int(gen);

	// Uniform distribution
	uniform_real_distribution<double> dist_double(0.0, max_weight);

	double beta = 0.0;

	// Draw new particles
	vector<Particle> new_particles;

	for (int i = 0; i < num_particles; i++) {
		beta += dist_double(gen) * 2.0;

		while (weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}

		new_particles.push_back(particles[index]);
	}

	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
