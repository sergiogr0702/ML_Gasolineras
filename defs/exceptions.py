"""
Custom exceptions
"""


class IllegalOperationError(Exception):
	pass


class NotEnoughDataError(Exception):
	"""
	Called when trying to run a model with a dataset that isn't large enough
	"""
	pass
