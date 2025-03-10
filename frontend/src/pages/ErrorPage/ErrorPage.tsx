import { useEffect } from "react";
import { useNavigate, useRouteError } from "react-router-dom";

import "./ErrorPage.css";
import Main from "@components/Main/Main";

interface RouteError {
  data: string;
  error: {
    columnNumber: number;
    fileName: string;
    lineNumber: number;
    message: string;
    stack: string;
  };
  internal: boolean;
  status: number;
  statusText: string;
}

const ErrorPage: React.FC = () => {
  const error = useRouteError() as RouteError;
  const navigate = useNavigate();

  useEffect(() => {
    setTimeout(() => {
      if (error.status === 404) {
        navigate("/", { replace: true });
      }
    }, 1000);
  }, [error, navigate]);

  return (
    <Main disableHeaderOffset>
      <h1>Упс! Что-то пошло не так.</h1>
      <p>Перенаправление на главную страницу...</p>
    </Main>
  );
};

export default ErrorPage;
