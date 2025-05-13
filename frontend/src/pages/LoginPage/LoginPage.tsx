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

import "./LoginPage.css";

const LoginPage: React.FC = () => {
  const sectionRef = useRef(null);

  const { isMobile, isSmallMobile } = useWindowSize();

  const {
    register,
    handleSubmit,
    formState: { errors, isValid, isSubmitted },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    mode: "onSubmit",
  });

  const handleLogin = async (data: LoginFormData) => {
    try {
      await axios.post("/api/login", data);

      console.log("Login successful");
    } catch (error) {
      console.error("Login failed:", error);
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
