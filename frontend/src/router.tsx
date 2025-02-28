import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { Home, Error } from "./pages";

const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <Home />,
      errorElement: <Error />,
    },
  ],
  { basename: import.meta.env.BASE_URL }
);

export default function Router() {
  return <RouterProvider router={router} />;
}
