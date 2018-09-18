from random import randint

from autokeras.net_transformer import transform
from autokeras.search import *


class RandomSearcher(Searcher):
    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        start_time = time.time()
        torch.cuda.empty_cache()
        model_id = self.model_count

        model_len = randint(3, 10)
        model_width = randint(32, 2048)
        graph = CnnGenerator(self.n_classes,
                             self.input_shape).generate(model_len,
                                                        model_width)
        # Start the new process for training.
        if self.verbose:
            print('\n')
            print('╒' + '=' * 46 + '╕')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('╘' + '=' * 46 + '╛')
        try:
            metric_value, loss, graph = train([graph, train_data, test_data, self.trainer_args,
                                               os.path.join(self.path, str(model_id) + '.png'),
                                               self.metric, self.loss, self.verbose])

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError

            self.add_model(metric_value, loss, graph, model_id, time.time() - start_time)

            pickle_to_file(self, os.path.join(self.path, 'searcher'))
            self.export_json(os.path.join(self.path, 'history.json'))

        except (mp.TimeoutError, TimeoutError) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('out of memory')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            pass


class GridSearcher(Searcher):
    def init_search(self):
        self.training_queue = []
        for model_len in range(3, 11):
            for model_width in [128, 256, 512, 1024, 2048]:
                self.training_queue.append((model_len, model_width))

    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        model_id = self.model_count

        if model_id >= len(self.training_queue):
            return

        model_len, model_width = self.training_queue[model_id]
        graph = CnnGenerator(self.n_classes,
                             self.input_shape).generate(model_len,
                                                        model_width)
        # Start the new process for training.
        if self.verbose:
            print('\n')
            print('╒' + '=' * 46 + '╕')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('╘' + '=' * 46 + '╛')
        try:
            metric_value, loss, graph = train([graph, train_data, test_data, self.trainer_args,
                                               os.path.join(self.path, str(model_id) + '.png'),
                                               self.metric, self.loss, self.verbose])

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError

            self.add_model(metric_value, loss, graph, model_id, time.time() - start_time)

            pickle_to_file(self, os.path.join(self.path, 'searcher'))
            self.export_json(os.path.join(self.path, 'history.json'))

        except (mp.TimeoutError, TimeoutError) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('out of memory')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            pass


class SeasSearcher(Searcher):
    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        Constant.N_NEIGHBOURS = 5
        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        # Start the new process for training.
        graph, father_id, model_id = self.training_queue.pop(0)
        if self.verbose:
            print('\n')
            print('╒' + '=' * 46 + '╕')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('╘' + '=' * 46 + '╛')
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(1)
        try:
            train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args,
                                                    os.path.join(self.path, str(model_id) + '.png'),
                                                    self.metric, self.loss, self.verbose)])

            # Do the search in current thread.
            searched = False
            new_graph = None
            new_father_id = None
            if not self.training_queue:
                searched = True
                best_graph = self.load_best_model()
                new_graphs = transform(best_graph)
                # Did not found a new architecture
                if len(new_graphs) == 0:
                    return
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((new_graph, new_father_id, new_model_id))
                self.descriptors.append(new_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError

            metric_value, loss, graph = train_results.get(timeout=remaining_time)[0]

            if self.verbose and searched:
                cell_size = [24, 49]
                header = ['Father Model ID', 'Added Operation']
                line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(header))
                print('\n' + '+' + '-' * len(line) + '+')
                print('|' + line + '|')
                print('+' + '-' * len(line) + '+')
                for i in range(len(new_graph.operation_history)):
                    if i == len(new_graph.operation_history) // 2:
                        r = [new_father_id, new_graph.operation_history[i]]
                    else:
                        r = [' ', new_graph.operation_history[i]]
                    line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(r))
                    print('|' + line + '|')
                print('+' + '-' * len(line) + '+')

            self.add_model(metric_value, loss, graph, model_id, time.time() - start_time)
            self.search_tree.add_child(father_id, model_id)
            self.bo.fit(self.x_queue, self.y_queue)
            self.x_queue = []
            self.y_queue = []

            pickle_to_file(self, os.path.join(self.path, 'searcher'))
            self.export_json(os.path.join(self.path, 'history.json'))

        except (mp.TimeoutError, TimeoutError) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('out of memory')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            pool.close()
            pool.join()


class BoSearcher(Searcher):
    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        # Start the new process for training.
        graph, father_id, model_id = self.training_queue.pop(0)
        graph.weighted = False
        if self.verbose:
            print('\n')
            print('╒' + '=' * 46 + '╕')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('╘' + '=' * 46 + '╛')
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(1)
        try:
            train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args,
                                                    os.path.join(self.path, str(model_id) + '.png'),
                                                    self.metric, self.loss, self.verbose)])

            # Do the search in current thread.
            searched = False
            new_graph = None
            new_father_id = None
            if not self.training_queue:
                searched = True
                new_graph, new_father_id = self.bo.optimize_acq(self.search_tree.adj_list.keys(),
                                                                self.descriptors,
                                                                timeout)
                # Did not found a new architecture
                if new_father_id is None:
                    return
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((new_graph, new_father_id, new_model_id))
                self.descriptors.append(new_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError

            metric_value, loss, graph = train_results.get(timeout=remaining_time)[0]

            if self.verbose and searched:
                cell_size = [24, 49]
                header = ['Father Model ID', 'Added Operation']
                line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(header))
                print('\n' + '+' + '-' * len(line) + '+')
                print('|' + line + '|')
                print('+' + '-' * len(line) + '+')
                for i in range(len(new_graph.operation_history)):
                    if i == len(new_graph.operation_history) // 2:
                        r = [new_father_id, new_graph.operation_history[i]]
                    else:
                        r = [' ', new_graph.operation_history[i]]
                    line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(r))
                    print('|' + line + '|')
                print('+' + '-' * len(line) + '+')

            self.add_model(metric_value, loss, graph, model_id, time.time() - start_time)
            self.search_tree.add_child(father_id, model_id)
            self.bo.fit(self.x_queue, self.y_queue)
            self.x_queue = []
            self.y_queue = []

            pickle_to_file(self, os.path.join(self.path, 'searcher'))
            self.export_json(os.path.join(self.path, 'history.json'))

        except (mp.TimeoutError, TimeoutError) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('out of memory')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            pool.close()
            pool.join()


class BfsSearcher(Searcher):
    def search(self, train_data, test_data, timeout=60 * 60 * 24):
        start_time = time.time()
        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        # Start the new process for training.
        graph, father_id, model_id = self.training_queue.pop(0)
        if self.verbose:
            print('\n')
            print('╒' + '=' * 46 + '╕')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('╘' + '=' * 46 + '╛')
        mp.set_start_method('spawn', force=True)
        pool = mp.Pool(1)
        try:
            train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args,
                                                    os.path.join(self.path, str(model_id) + '.png'),
                                                    self.metric, self.loss, self.verbose)])

            # Do the search in current thread.
            searched = False
            new_graph = None
            new_father_id = None
            if not self.training_queue:
                searched = True
                new_graph, new_father_id = self.bo.optimize_acq(self.search_tree.adj_list.keys(),
                                                                self.descriptors,
                                                                timeout)
                new_graphs =
                # Did not found a new architecture
                if new_father_id is None:
                    return
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((new_graph, new_father_id, new_model_id))
                self.descriptors.append(new_graph.extract_descriptor())

            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                raise TimeoutError

            metric_value, loss, graph = train_results.get(timeout=remaining_time)[0]

            if self.verbose and searched:
                cell_size = [24, 49]
                header = ['Father Model ID', 'Added Operation']
                line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(header))
                print('\n' + '+' + '-' * len(line) + '+')
                print('|' + line + '|')
                print('+' + '-' * len(line) + '+')
                for i in range(len(new_graph.operation_history)):
                    if i == len(new_graph.operation_history) // 2:
                        r = [new_father_id, new_graph.operation_history[i]]
                    else:
                        r = [' ', new_graph.operation_history[i]]
                    line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(r))
                    print('|' + line + '|')
                print('+' + '-' * len(line) + '+')

            self.add_model(metric_value, loss, graph, model_id, time.time() - start_time)
            self.search_tree.add_child(father_id, model_id)
            self.bo.fit(self.x_queue, self.y_queue)
            self.x_queue = []
            self.y_queue = []

            pickle_to_file(self, os.path.join(self.path, 'searcher'))
            self.export_json(os.path.join(self.path, 'history.json'))

        except (mp.TimeoutError, TimeoutError) as e:
            raise TimeoutError from e
        except RuntimeError as e:
            if not re.search('out of memory', str(e)):
                raise e
            if self.verbose:
                print('out of memory')
            Constant.MAX_MODEL_SIZE = graph.size() - 1
            return
        finally:
            # terminate and join the subprocess to prevent any resource leak
            pool.close()
            pool.join()
