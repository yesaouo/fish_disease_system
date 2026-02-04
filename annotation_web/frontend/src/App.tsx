import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { DatasetProvider, useDataset } from "./context/DatasetContext";
import LoginPage from "./features/auth/LoginPage";
import DatasetPickerPage from "./features/datasets/DatasetPickerPage";
import AnnotationPage from "./features/annotation/AnnotationPage";
import AdminDashboard from "./features/admin/AdminDashboard";
import AnnotatedListPage from "./features/annotated/AnnotatedListPage";
import CommentedListPage from "./features/commented/CommentedListPage";
import HealthyImagesPage from "./features/healthy/HealthyImagesPage";
import HealthyAnnotationPage from "./features/healthy/HealthyAnnotationPage";

const RequireAuth: React.FC<{ children: React.ReactElement }> = ({
  children
}) => {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

const RequireDataset: React.FC<{ children: React.ReactElement }> = ({
  children
}) => {
  const { dataset } = useDataset();
  if (!dataset) {
    return <Navigate to="/datasets" replace />;
  }
  return children;
};

const AppRoutes = () => {
  const { isAuthenticated } = useAuth();
  const { dataset } = useDataset();
  const defaultPath = isAuthenticated
    ? dataset
      ? "/annotate"
      : "/datasets"
    : "/login";

  return (
    <Routes>
      <Route path="/" element={<Navigate to={defaultPath} replace />} />
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/datasets"
        element={
          <RequireAuth>
            <DatasetPickerPage />
          </RequireAuth>
        }
      />
      <Route
        path="/annotate"
        element={
          <RequireAuth>
            <RequireDataset>
              <AnnotationPage />
            </RequireDataset>
          </RequireAuth>
        }
      />
      <Route
        path="/annotate/:index"
        element={
          <RequireAuth>
            <RequireDataset>
              <AnnotationPage />
            </RequireDataset>
          </RequireAuth>
        }
      />
      <Route
        path="/annotated"
        element={
          <RequireAuth>
            <RequireDataset>
              <AnnotatedListPage />
            </RequireDataset>
          </RequireAuth>
        }
      />
      <Route
        path="/commented"
        element={
          <RequireAuth>
            <RequireDataset>
              <CommentedListPage />
            </RequireDataset>
          </RequireAuth>
        }
      />
      <Route
        path="/healthy"
        element={
          <RequireAuth>
            <RequireDataset>
              <HealthyImagesPage />
            </RequireDataset>
          </RequireAuth>
        }
      />
      <Route
        path="/healthy/:index"
        element={
          <RequireAuth>
            <RequireDataset>
              <HealthyAnnotationPage />
            </RequireDataset>
          </RequireAuth>
        }
      />
      <Route
        path="/admin"
        element={
          <RequireAuth>
            <AdminDashboard />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to={defaultPath} replace />} />
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <DatasetProvider>
        <AppRoutes />
      </DatasetProvider>
    </AuthProvider>
  );
};

export default App;
