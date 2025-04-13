import { createBrowserRouter, RouterProvider } from "react-router-dom";
import Home from "./pages/Home";
import Dashboard from "./pages/dashboard";
import Scan from "./pages/scan";
import Auth from "./pages/Auth";
import Chatbox from "./pages/Chatbox";
import ResultsPage from "./pages/Results";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/dashboard",
    element: <Dashboard />,
  },
  {
    path: "/scan",
    element: <Scan />,
  },
  {
    path: "/auth",
    element: <Auth />,
  },
  {
    path: "/chatbox",
    element: <Chatbox />,
  },
  {
    path: "/result",
    element: <ResultsPage />,
  },
]);

export default function App() {

  return <RouterProvider router={router} /> ;
}
