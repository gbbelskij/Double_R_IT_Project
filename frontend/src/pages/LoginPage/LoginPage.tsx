import { useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import axios from "axios";

import { loginSchema, LoginFormData } from "@schemas/loginSchema";

import { useWindowSize } from "@hooks/useWindowSize";

import Main from "@components/Main/Main";
import Form from "@components/Form/Form";
import Input from "@components/Input/Input";
import Checkbox from "@components/Checkbox/Checkbox";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { DAY, MONTH } from "@data/constants";
import { useNavigate } from "react-router";

const LoginPage: React.FC = () => {
  const sectionRef = useRef(null);
  const navigate = useNavigate();

  const { isMobile, isSmallMobile } = useWindowSize();

  const {
    register,
    handleSubmit,
    formState: { errors, isValid, isSubmitted },
    setError,
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    mode: "onSubmit",
  });

  const handleLogin = async (data: LoginFormData) => {
    const { remember, ...formDataToSend } = data;

    try {
      const response = await axios.post("/api/login", formDataToSend);

      const jwtToken = response.data.token;
      const maxAge = remember ? MONTH : DAY;

      document.cookie = `token=${jwtToken}; max-age=${maxAge}; path=/; secure; samesite=strict`;

      navigate("/");
    } catch (error) {
      let errorMessage = "Что-то пошло не так. Попробуйте позже.";

      if (axios.isAxiosError(error) && error.response?.data?.message) {
        errorMessage = error.response.data.message;

        if (errorMessage === "Invalid password") {
          setError("password", {
            type: "manual",
            message: "Неверный пароль",
          });

          return;
        }

        if (errorMessage === "User not found") {
          setError("email", {
            type: "manual",
            message: "Пользователь не найден",
          });

          return;
        }
      }

      navigate("/error", {
        state: {
          errorHeading: "Ошибка входа",
          errorText: errorMessage,
          timeout: 0,
        },
      });
    }
  };

  const isButtonDisabled =
    isSubmitted && (!isValid || Object.keys(errors).length > 0);

  return (
    <Main disableHeaderOffset>
      <Form
        id="login-form"
        title="Вход"
        actionText="Войти"
        helperText="У вас ещё нет аккаунта?"
        helperLink="/registration"
        helperLinkText="Зарегистрируйтесь!"
        handleAction={handleSubmit(handleLogin)}
        logoOffset={isSmallMobile ? 30 : isMobile ? 40 : 50}
        logoAlign="end"
        isButtonDisabled={isButtonDisabled}
        ref={sectionRef}
      >
        <Input
          type="email"
          name="email"
          label="Эл. почта"
          register={register}
          error={errors.email}
        />
        <Input
          type="password"
          name="password"
          label="Пароль"
          register={register}
          error={errors.password}
        />
        <Checkbox
          name="remember"
          label="Запомнить меня"
          register={register}
          error={errors.remember}
        />
      </Form>

      <BackgroundElements targetRef={sectionRef} />
    </Main>
  );
};

export default LoginPage;
