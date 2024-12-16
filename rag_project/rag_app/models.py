from django.db import models

class Query(models.Model):
    prompt = models.TextField(verbose_name='질문')
    response = models.TextField(verbose_name='응답')
    sources = models.JSONField(verbose_name='출처')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일시')

    class Meta:
        ordering = ['-created_at']
        verbose_name = '질의'
        verbose_name_plural = '질의들'