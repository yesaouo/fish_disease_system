import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { DatasetProvider } from "./context/DatasetContext";
import LoginPage from "./features/auth/LoginPage";
import HomePage from "./features/home/HomePage";
import DatasetPickerPage from "./features/datasets/DatasetPickerPage";
import AnnotationPage from "./features/annotation/AnnotationPage";
import AdminDashboard from "./features/admin/AdminDashboard";
import AnnotatedListPage from "./features/annotated/AnnotatedListPage";
import CommentedListPage from "./features/commented/CommentedListPage";
import HealthyImagesPage from "./features/healthy/HealthyImagesPage";
import HealthyAnnotationPage from "./features/healthy/HealthyAnnotationPage";
import DiagnosisPage from "./features/diagnosis/DiagnosisPage";

const RequireAuth: React.FC<{ children: React.ReactElement }> = ({
  children
}) => {
  const { isAuthenticated } = useAuth();
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

const RequireEditor: React.FC<{ children: React.ReactElement }> = ({
  children
}) => {
  const { canEdit } = useAuth();
  if (!canEdit) {
    return <Navigate to="/datasets" replace />;
  }
  return children;
};

const AppRoutes = () => {
  const { isAuthenticated } = useAuth();
  // 獨立首頁為登入後的落地頁；不再依 dataset 自動跳進標註。
  const defaultPath = isAuthenticated ? "/home" : "/login";

  return (
    <Routes>
      <Route path="/" element={<Navigate to={defaultPath} replace />} />
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/home"
        element={
          <RequireAuth>
            <HomePage />
          </RequireAuth>
        }
      />
      <Route
        path="/datasets"
        element={
          <RequireAuth>
            <DatasetPickerPage />
          </RequireAuth>
        }
      />
      <Route
        path="/annotate/:dataset/new"
        element={
          <RequireAuth>
            <RequireEditor>
              <AnnotationPage />
            </RequireEditor>
          </RequireAuth>
        }
      />
      <Route
        path="/annotate/:dataset"
        element={
          <RequireAuth>
            <AnnotationPage />
          </RequireAuth>
        }
      />
      <Route
        path="/annotate/:dataset/:index"
        element={
          <RequireAuth>
            <AnnotationPage />
          </RequireAuth>
        }
      />
      <Route
        path="/annotated/:dataset"
        element={
          <RequireAuth>
            <AnnotatedListPage />
          </RequireAuth>
        }
      />
      <Route
        path="/commented/:dataset"
        element={
          <RequireAuth>
            <CommentedListPage />
          </RequireAuth>
        }
      />
      <Route
        path="/healthy/:dataset"
        element={
          <RequireAuth>
            <HealthyImagesPage />
          </RequireAuth>
        }
      />
      <Route
        path="/healthy/:dataset/:index"
        element={
          <RequireAuth>
            <HealthyAnnotationPage />
          </RequireAuth>
        }
      />
      <Route
        path="/diagnose"
        element={
          <RequireAuth>
            <DiagnosisPage />
          </RequireAuth>
        }
      />
      <Route
        path="/admin"
        element={
          <RequireAuth>
            <RequireEditor>
              <AdminDashboard />
            </RequireEditor>
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
