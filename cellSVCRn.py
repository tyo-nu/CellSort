__author__ = 'Dante'

import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler as stds
from sklearn.grid_search import GridSearchCV as GSC
from sklearn.cross_validation import train_test_split as tts
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import itertools
from argparse import ArgumentParser
from FlowCytometryTools import FCMeasurement


def pre_process(cFile, points=500, channels=('FITC-A', 'PE-Texas Red-A'), fcs=True):
	"""Pre-processes datasets to usable forms."""

	text_name = cFile.split('.')[0] + ".txt"

	if fcs:
		sample = FCMeasurement(ID="sample", datafile=cFile)
		allowed = list(sample.meta['_channel_names_'])

		for chan in channels:
			if chan not in allowed:
				raise ValueError('%s is not a valid channel. Valid channels for %s include:\n%s' % (chan, cFile, str(allowed)))

		data_columns = (sample.data[meas].values for meas in channels)
		np.savetxt(text_name, np.column_stack(data_columns), delimiter='\t', fmt="%.2f")

	#Open file and check for conformation to format.
	f = open(text_name, 'r')
	fData = f.readlines()
	f.close()

	if len(fData[0].strip('\n').split('\t')) != len(channels):
		raise IOError('Input files must be %s-column, tab-delimited text files.' % str(len(channels)))

	#Parse data files, generate random points.
	signals = {n: np.array([float(x.strip('\n').split('\t')[n]) for x in fData]) for n in range(len(channels))}
	indices = random.sample(range(signals[0].size), points)
	randsignals = {k: v[indices] for k, v in signals.iteritems()}

	#Add to array.
	sig_values = tuple(randsignals.values())
	darray = np.vstack(sig_values).T

	return darray


def enrichment(a, b, p, i, N=10000000):
	"""Calculates the rounds of enrichment required given the data"""

	pPos = (p * (1 - b)) / ((p * (1 - b)) + (a * (1 - p)))

	print "Round %s:\tP(+|D) = %s\t%s discarded" % (i, round(pPos, 4), str(round(100 * ((N * (1 - pPos)) / N), 4)) + '%')

	return pPos


def _scale(neg_x, pos_x, neg_y, pos_y):
	#Instantiate a scaler class, makes data mean = 0, stddev = 1.
	scaler = stds().fit(neg_x)
	neg_scale = scaler.transform(neg_x)
	pos_scale = scaler.transform(pos_x)

	x = np.vstack((neg_scale, pos_scale))
	y = np.hstack((neg_y, pos_y))

	return scaler, x, y

def _unscale(scaler, xx, yy, b_):
	#De-scaling routine for translated parameters.
	xx_unscale = (scaler.std_[0] * xx) + scaler.mean_[0]
	yy_unscale = (scaler.std_[1] * yy) + scaler.mean_[1]
	m_unscale = (yy_unscale[1] - yy_unscale[0]) / (xx_unscale[1] - xx_unscale[0])
	b_unscale = (scaler.std_[1] * b_) + scaler.mean_[1]

	return xx_unscale, yy_unscale, m_unscale, b_unscale


class Equation():
	def __init__(self, xo, yo, b):
		self.x_ = xo
		self.y_ = yo
		self.b_ = b

		self.slope = -self.x_ / self.y_
		self.intercept = -self.b_ / self.y_

	def xy_pairs(self, x_seq):
		#Generate x-y pairs for the hyperplane.
		yy = self.slope * x_seq + self.intercept

		return yy

	def test(self, x, y):
		if y >= self.slope * x + self.intercept:
			return 1
		else:
			return 0


	def display(self):
		"""Displays the scaled equation."""
		print "y = %f * x + %f" % (self.slope, self.intercept)
		return self.slope, self.intercept


class classifier():
	def __init__(self, negative, positive, pts=500, channels=('FITC-A', 'PE-Texas Red-A'), fcs=True):
		self.neg = pre_process(negative, pts, channels=channels, fcs=fcs)
		self.pos = pre_process(positive, pts, channels=channels, fcs=fcs)

		if self.neg.shape[1] != self.pos.shape[1]:
			raise IOError("Control sets must have the same number of channels of data")
		if self.neg.shape[1] != len(channels):
			raise IOError("Number of columns in datafiles does not match the number of labels specified. Use the -L flag to specify a list of labels.")

		self.neg_y = -1 * np.ones(self.neg.shape[0])
		self.pos_y = np.ones(self.pos.shape[0])

		self.channels = {k: v for k, v in enumerate(channels)}

		self.scaler, self.x, self.y = _scale(self.neg, self.pos, self.neg_y, self.pos_y)


	def plot_2d_hyperplane(self, plottables_, save=False):
		#Plotting routine.
		xx_unscale = plottables_[0][1]
		yy_unscale = plottables_[1][1]

		x_chan = plottables_[0][0]
		y_chan = plottables_[1][0]

		i_pos = [i for i, x in enumerate(self.y_eval) if x == 1]
		i_neg = [i for i, x in enumerate(self.y_eval) if x == -1]

		plt.rcParams['font.sans-serif'] = ['Arial']

		plt.tick_params(labelsize=16)
		plt.xticks(rotation=45)
		plt.xlabel(self.channels[x_chan], size=18, labelpad=14)
		plt.ylabel(self.channels[y_chan], size=18, labelpad=14)
		plt.xlim(-50000, 200000)
		plt.ylim(-25000, 75000)
		# Block will plot scaled data; deprecate in future.
		# plt.scatter(x_eval[:, 0], x_eval[:, 1], c=y_eval, cmap=plt.cm.Paired)
		# plt.plot(xx, yy, 'k-')
		# plt.ylim(-1, 10)
		# plt.savefig('scaled_hyp.svg')
		# plt.show()
		# plt.cm.Paired is a matplotlib builtin colormap.  Helpfully, it isn't documented anywhere in the matplotlib
		# API.
		#plt.scatter(self.x_eval_unscale[:, x_chan], self.x_eval_unscale[:, y_chan], c=self.y_eval, cmap=plt.cm.Paired)
		#plt.colorbar()
		p = plt.scatter(self.x_eval_unscale[i_pos, x_chan], self.x_eval_unscale[i_pos, y_chan], marker='o', color='b')
		n = plt.scatter(self.x_eval_unscale[i_neg, x_chan], self.x_eval_unscale[i_neg, y_chan], marker='x', color='r')
		plt.legend((p, n), ('Positive Control', 'Negative Control'), scatterpoints=1, loc=0)
		plt.plot(xx_unscale, yy_unscale, 'k-', linewidth=2)
		if save:
			plt.savefig('unscaled_hyp_%s_%s.svg' % (self.channels[y_chan], self.channels[x_chan]))
		plt.show()


	def classify(self, C=None, cv=5, test_size=0.90, weight=(1, 1), shannon=0.95, validate_off=True):
		#Outputs necessary data for the hyperplane. In simple RFP/GFP experiments, this is a line.
		#Split data for training, and scale.
		self.x_dev, self.x_eval, self.y_dev, self.y_eval = tts(self.x, self.y, test_size=test_size, random_state=0)
		self.i_pos = [i for i, x in enumerate(self.y_eval) if x == 1]
		self.i_neg = [i for i, x in enumerate(self.y_eval) if x == -1]
		wt = {-1: weight[0], 1: weight[1]}

		#Parameter estimation routine.
		if C is None:
			parameter_space = {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 2, 5, 10]}
			grid = GSC(SVC(), parameter_space, cv=cv, scoring='f1')
			grid.fit(self.x_dev, self.y_dev)
			c = grid.best_params_['C']
		else:
			c = C

		#Fitting and cross-validation to recover FPR, FNR, and hyperplane parameters.
		clf = SVC(kernel='linear', C=c, class_weight=wt)
		kf = KFold(len(self.y_eval), n_folds=5)
		_c_matrices = []
		_w = []
		_intercepts = []
		_auc = []

		for train, test in kf:
			clf.fit(self.x_eval[train], self.y_eval[train])
			y_predict = clf.predict(self.x_eval[test])
			_c_matrices.append(confusion_matrix(self.y_eval[test], y_predict))
			_w.append(clf.coef_[0])
			_intercepts.append(clf.intercept_[0])
			_auc.append(roc_auc_score(self.y_eval[test], y_predict))

		c_matrix = np.sum(np.dstack(tuple(_c_matrices)), 2) / len(_c_matrices)
		fpr = float(c_matrix[0, 1]) / (float(c_matrix[0, 1]) + float(c_matrix[0, 0]))
		fnr = float(c_matrix[1, 0]) / (float(c_matrix[1, 0]) + float(c_matrix[1, 1]))

		if validate_off:

			#Calculate FPR, FNR, hyperplane parameters
			w = np.sum(np.vstack(tuple(_w)), 0) / len(_w)
			intercept = np.sum(_intercepts) / len(_intercepts)

			#Unscale data for plotting, generate points on the hyperplane, instantiate exportable dict.
			self.x_eval_unscale = self.scaler.inverse_transform(self.x_eval)
			xx = np.linspace(-15, 30)
			plottables = []
			self.ent_ = []

			#Iterate through n-choose-2 hyperplanes, where n = number of features.
			for pair in itertools.combinations(range(self.x_eval.shape[1]), 2):
				hyp = Equation(w[pair[0]], w[pair[1]], intercept)
				yy = hyp.xy_pairs(xx)
				xx_unscale, yy_unscale, m_unscale, b_unscale = _unscale(self.scaler, xx, yy, hyp.intercept)
				eq_ = "%s(%s) = %f * %s + %f" % (self.channels[pair[1]], self.channels[pair[0]], m_unscale, self.channels[pair[0]], b_unscale)
				plot_ = [(pair[0], xx_unscale), (pair[1], yy_unscale)]
				plottables.append(plot_)

				over = [i for i, row in enumerate(self.x_eval_unscale[:, pair]) if row[1] >= m_unscale * row[0] + b_unscale]
				under = [i for i, row in enumerate(self.x_eval_unscale[:, pair]) if row[1] < m_unscale * row[0] + b_unscale]

				try:
					p_over = float(sum(self.y_eval[over] == 1)) / float(self.y_eval[over].size)
					H_over = -(p_over * math.log(p_over, 2)) - ((1 - p_over) * (math.log((1 - p_over), 2)))
				except (ZeroDivisionError, ValueError):
					H_over = 0.0

				try:
					p_under = float(sum(self.y_eval[under] == 1)) / float(self.y_eval[under].size)
					H_under = -(p_under * math.log(p_under, 2)) - ((1 - p_under) * (math.log((1 - p_under), 2)))
				except (ZeroDivisionError, ValueError):
					H_under = 0.0

				p_pts_over = float(self.y_eval[over].size) / float(self.y_eval[over].size + self.y_eval[under].size)
				p_pts_under = float(self.y_eval[under].size) / float(self.y_eval[over].size + self.y_eval[under].size)

				H_ = p_pts_over * H_over + p_pts_under * H_under

				self.ent_.append((H_, eq_))

			print "Use the following gates:"

			for h, l in self.ent_:
				if h < shannon:
					print l, "H = %f" % h

			return fpr, fnr, plottables, _auc

		else:
			return fpr, fnr, _auc


if __name__ == "__main__":

	desc = "Analyze control datasets and infer a separating hyperplane that will accurately separate the experimental group"
	parser = ArgumentParser(description=desc)
	parser.add_argument('neg_ctrl', help="negative control data")
	parser.add_argument('pos_ctrl', help="positive control data")
	parser.add_argument('-c', '--cellstosort', dest='cells', default=10000000, help="number of cells sorted")
	parser.add_argument('-C', '--C_parameter', dest='C', default=1.0, help="SVC soft margin parameter value")
	parser.add_argument('-e', '--estimate_C', dest='estimate', action='store_true', default=False)
	parser.add_argument('-f', '--fcsformat', dest='format', action='store_false', default=True, help="Set if inputting txt files directly")
	parser.add_argument('-g', '--graph', dest='graph', action='store_true', default=False)
	parser.add_argument('-H', '--entropy', dest='shannon', default=0.95, help="Threshold value for hyperplane utility")
	parser.add_argument('-L', '--labels', dest='labels', nargs='*', default=['FITC-A', 'PE-Texas Red-A'],
	                  help='channel/column labels, in the order that they appear in the text files. Must ALWAYS be last arg.')
	parser.add_argument('-p', '--probability', dest='prob', default=0.000001,
	                  type=float, help="proportion of good cells in initial population")
	parser.add_argument('-r', '--replicates', dest='repeat', default=0, type=int, help='number of runs in validation')
	parser.add_argument('-s', '--savegraph', dest='save', action='store_true', default=False)
	parser.add_argument('-t', '--training_points', dest='training', default=5000,
	                  help="number of training examples to use")
	parser.add_argument('-w', '--weight_classes', nargs=2, dest='weight', default=(1, 1),
	                  help="class weights stored as {-1: _, 1: _}")
	parser.add_argument('-x', '--cross_validations', dest='xval', type=int, default=5)
	args = parser.parse_args()

	channels = tuple(args.labels)
	w = args.weight

	if args.repeat == 0:
		s = classifier(args.neg_ctrl, args.pos_ctrl, pts=args.training, channels=channels, fcs=args.format)

		if args.shannon >= 1 or args.shannon <= 0:
			raise ValueError('Shannon entropy must be a value in the open interval (0, 1).')

		if args.estimate:
			alpha, beta, plotinfo, auc = s.classify(weight=w, cv=args.xval, shannon=args.shannon)
		else:
			alpha, beta, plotinfo, auc = s.classify(C=args.C, weight=w, cv=args.xval, shannon=args.shannon)

		#print auc

		if args.graph and len(plotinfo) > 0:
			for p in plotinfo:
				s.plot_2d_hyperplane(p, save=args.save)

		pPos_init = args.prob
		pPos = pPos_init

		print "________________________________________________"
		print "P(D|-) = %s\tP(D|+) = %s\tAUC = %s" % (str(round(alpha, 4)), str(round(1 - beta, 4)), str(np.mean(np.array(auc))))

		i = 1
		while pPos <= 0.99:
			pPos = enrichment(alpha, beta, pPos, i, N=args.cells)
			i += 1
	else:
		auc_results = []
		fpr_results = []
		fnr_results = []
		while len(auc_results) < args.repeat:
			s = classifier(args.neg_ctrl, args.pos_ctrl, pts=args.training, channels=channels)
			if args.estimate:
				fpr, fnr, auc = s.classify(weight=w, cv=args.xval, shannon=args.shannon, validate_off=False)
			else:
				fpr, fnr, auc = s.classify(C=args.C, weight=w, cv=args.xval, shannon=args.shannon, validate_off=False)

			auc_results.append(float(sum(auc)) / float(len(auc)))
			fpr_results.append(fpr)
			fnr_results.append(fnr)

		print float(sum(auc_results)) / float(len(auc_results)), np.std(np.array(auc_results))
		print float(sum(fpr_results)) / float(len(fpr_results)), np.std(np.array(fpr_results))
		print float(sum(fnr_results)) / float(len(fnr_results)), np.std(np.array(fnr_results))
