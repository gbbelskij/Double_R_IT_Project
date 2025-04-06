import { useEffect } from "react";
import { useNavigate, useRouteError } from "react-router-dom";

import Main from "@components/Main/Main";
import Logo from "@components/Logo/Logo";

import "./ErrorPage.css";

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
      <div className="error-container">
        <Logo />

        <h1>Упс! Что-то пошло не так!</h1>

        <p>Перенаправление на главную страницу...</p>
      </div>
    </Main>
  );
};

export default ErrorPage;
