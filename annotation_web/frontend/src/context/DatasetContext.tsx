import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState
} from "react";

type DatasetContextValue = {
  dataset: string | null;
  classes: string[];
  setDataset: (dataset: string | null) => void;
  setClasses: (classes: string[]) => void;
};

const DATASET_KEY = "annotatorDataset";

const DatasetContext = createContext<DatasetContextValue | null>(null);

export const DatasetProvider: React.FC<{ children: React.ReactNode }> = ({
  children
}) => {
  const [dataset, setDatasetState] = useState<string | null>(() => {
    return localStorage.getItem(DATASET_KEY);
  });
  const [classes, setClassesState] = useState<string[]>([]);

  const setDataset = useCallback((next: string | null) => {
    setDatasetState(next);
    if (next) {
      localStorage.setItem(DATASET_KEY, next);
    } else {
      localStorage.removeItem(DATASET_KEY);
    }
  }, []);

  const setClasses = useCallback((next: string[]) => {
    setClassesState(next);
  }, []);

  const value = useMemo(
    () => ({
      dataset,
      classes,
      setDataset,
      setClasses
    }),
    [dataset, classes, setDataset, setClasses]
  );

  return (
    <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>
  );
};

export const useDataset = (): DatasetContextValue => {
  const ctx = useContext(DatasetContext);
  if (!ctx) {
    throw new Error("useDataset must be used within DatasetProvider");
  }
  return ctx;
};

